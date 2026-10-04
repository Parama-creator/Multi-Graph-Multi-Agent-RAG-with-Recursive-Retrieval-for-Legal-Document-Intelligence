# Recursive Retrieval on a Lexical Graph: a RAG pipeline for legal documents

A retrieval-augmented generation (RAG) pipeline for long, heavily cross-referenced documents such as regulations, policies and compliance frameworks. Instead of treating a document as a flat list of text chunks, it turns the document into a **graph** and lets a set of LLM agents **follow references** ("see para 7.3", footnotes, hyperlinks) until enough context has been gathered to answer.

---

## 1. The problem

Standard RAG splits a document into chunks, embeds them, and returns the top-k most similar ones. That breaks down on legal and regulatory text, for four reasons:

| Failure | Example |
|---|---|
| **Clauses are split apart** | A hit on sub-clause 7.2(b) returns without its parent clause 7.2, so the meaning is incomplete. |
| **Answers live behind cross-references** | Para 6.3 says "refer to Para 7.3 and 7.4", and 7.3 in turn mentions Para 9.1. Similarity search won't surface them because they don't resemble the question. |
| **Footnotes carry obligations** | The relevant condition sits in a footer that refers to another section. |
| **Defined terms are used without explanation** | "CCO" or "control function" only make sense through the definitions page. |

This pipeline addresses each of those by modelling document structure and links explicitly.

**Target behaviour on the original test question** ("How can the Board and the CCO manage control functions?"):
1. Give the definition of CCO from the definitions page.
2. Retrieve the clauses in 6.3 and 7.2.
3. Detect "Refer to Para 7.3 and 7.4".
4. Pull in Paras 7.3 and 7.4, which mention Para 9.1, and pull that in too.

---

## 2. Pipeline at a glance

```
                          ┌──────────────────────────────── INDEXING (once per document) ───────────────────────────────┐
  PDF ──► pymupdf ──► hyperlinks (from → to coordinates)                                                                  │
  PDF ──► Reducto ──► layout elements (type, text, page, bounding box)                                                    │
                          │                                                                                               │
                          ├─ match each link end to the nearest Reducto element (geometry)  ──► (source, destination) pairs│
                          ├─ GPT-4o on definition pages ──► Definitions graph:  Term ─has_definition─► Definition          │
                          └─ elements + pairs ──► Lexical graph: nodes (+ embeddings) and contains/is_parent/follows/links_to
                          │                                                                                               │
                          └─ LlamaIndex: vector index + keyword index + BM25 over all lexical-graph nodes                  │
                          └───────────────────────────────────────────────────────────────────────────────────────────────┘

                          ┌──────────────────────────────── QUERY TIME (LangGraph agents) ──────────────────────────────┐
  question ─► Initial Search ─► Definition Agent ─► Context Fetch ─► Footer Parsing ─► Supervisor ─► Router ─┐            │
                (hybrid + prune)                    (full clauses +    (footnote        (continue /    (which links / │
                                                     link hints)        context)         end)           footers?)     │
                                       ▲                                                                       │       │
                                       └──────────────── Recursive Retrieval (fetch those nodes) ◄─────────────┘       │
                                                                                                                         │
                                                          nothing left to explore ─► Answering Agent ─► answer          │
                          └───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Indexing

### 3.1 Document parsing (Reducto)
[Reducto](https://reducto.ai) parses the PDF into layout elements. Each element has a `type`, `content`, and a `bbox` (page number and normalized `left/top/width/height`).

Element types kept: `Title`, `Section Header`, `Text`, `List Item`, `Table`, `Figure`, `Header`, `Footer`. Others (for example `Page Number`) are skipped.

List items keep their tab characters, which is how the indentation depth of sub-clauses is detected.

### 3.2 Link detection (pymupdf)
Hyperlinks in a PDF are stored as geometry (a rectangle on one page pointing to a point on another page), not as element IDs. Reducto elements are also geometry. The pipeline connects the two:

1. `pymupdf` lists every link on every page, with its source rectangle and, for internal links, the destination page and point.
2. Both points are normalized by the PDF page size and compared with Reducto bounding boxes (normalized by Reducto's own page size, with a vertical stretch of `0.96` that was needed to match Reducto's output).
3. The **closest Reducto element** to the link's source becomes the source node; the closest element to the destination point becomes the destination node.
4. The result is a list of `(source_element_hash, destination_element_hash)` pairs. External URL links have no destination node and are dropped.

### 3.3 Lexical graph
A `networkx.MultiDiGraph`. Every Reducto element becomes a node.

**Node ID:** first 50 characters of the content + `_` + MD5 hash of (content + bbox). This keeps IDs unique, readable in logs, and informative to the LLM, which sees the ID and can judge whether following it is worthwhile.

**Node attributes:** `label` (element type), `text`, `properties` (Type, Hash, Node Name, Content, Page Number, Location), and `embedding`.

**Edges** (the relation name is the edge key):

| Relation | Meaning | Built by |
|---|---|---|
| `contains` | A list item contains its indented sub-items; the document contains section headers and cover-page elements | Indentation depth (tab count) vs. the last top-level list item |
| `is_parent` | A section is the parent of its text, tables and top-level list items | Last seen Section Header |
| `follows` | Section header → next section header (document flow) | Order of headers |
| `links_to` | A hyperlink from one element to another | Section 3.2 |

Extra nodes: a `Document` node (the file name) and a `Cover Page` node. Everything on page 1 hangs off the cover page; elements before the first section header attach to the document.

**Embeddings:** every node's text is embedded with an OpenAI model (`text-embedding-3-small` by default; texts truncated to 20,000 characters, empty strings replaced). The model used at indexing **must** be the one used to embed queries.

### 3.4 Definitions graph
GPT-4o reads the definition pages and extracts `(term, definition)` pairs verbatim using structured output. Pairs are stored in a small `DiGraph`:

```
Term ──has_definition──► Definition
```

### 3.5 Retrieval indexes (LlamaIndex)
Each lexical-graph node is wrapped in `MultiAgentSearchLocalNode`, a `TextNode` subclass that carries the element type, embedding, metadata, **children** (to represent clause trees) and **context** (hints added during retrieval). It can print itself as a nested prompt for the LLM. Three indexes are built over these nodes:

- **Vector index** (semantic similarity, using the precomputed embeddings)
- **Keyword table index** (simple keyword extraction)
- **BM25** (stemmed, English stopwords)

---

## 4. Query time: the agent workflow

The workflow is a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine. All agents share one `AgentState`:

`query` · `definitions` · `previous_nodes` · `last_fetched_context_nodes` · `node_links_to_fetch` · `node_footers_to_fetch` · `allow_continue` · `search_failures` · `markdown_debug` · `pass_count`

### 4.1 Initial Search Agent
- Runs a **hybrid retriever** (`VectorBM25`): 1 vector hit + 7 BM25 hits, merged with OR (union).
- A GPT-4o **pruning** step then removes nodes that are not relevant to the query, each with a short reason. (Pruning a node also removes its children.)

### 4.2 Definition Agent
- GPT-4o is shown the list of all defined terms and the question, and returns **only which terms are relevant**.
- Definitions are then read **verbatim from the graph**, so they cannot be hallucinated. Matching is case-insensitive.

### 4.3 Context Fetch Tool
Two deterministic graph operations, no LLM involved:

**Whole-clause expansion** (`fetch_whole_clause_lists`). A hit may be a clause title or just one sub-clause:
```
CLAUSE 1                       Input:  [Section Header, Text, List Item(Subclause 1.2), Figure]
 - Clause Title 1              Output: [Section Header, Text, List Item(Clause 1 Title), Figure]
    - Subclause 1.1                                              └── children: [1.1, 1.2]
    - Subclause 1.2
```
- If a list item is contained by another list item, that parent is the clause title (the hit was a sub-clause). Otherwise the hit itself is a clause title.
- All of the clause's list-item children are then attached, and each full clause is returned only once.
- Depth is assumed to be two levels (clause and sub-clause).

**Link hints** (`fill_nodes_with_link_hints`). For each node with outgoing `links_to` edges, a `link` entry is added to its context listing the destination node IDs. The LLM then sees something like "this clause links to: `[7.3 Control functions…_ab12…]`" and, because IDs contain a content preview, can decide whether to follow it.

### 4.4 Footer Parsing Agent
For each page represented in the current nodes, GPT-4o receives that page's nodes and that page's footers and returns:
- the footer text relevant to each node (matched through footnote numbering), and
- a flag `further_explore_footer` when the footer points to another part of the *same* document.

Results are stored in each node's context as `footer info` and `further explore footer?`. If this step fails, a search failure is logged and the process continues.

### 4.5 Supervisor Agent
- Reads all nodes gathered so far and the list of search failures. If there are many repeated failures with nothing new being found, it answers **END** (go to the Answering Agent); otherwise **CONTINUE**.
- Guards the context size: if the gathered nodes exceed the **32k-token** budget, it prunes with the LLM and logs whether the budget was met.

### 4.6 Router Agent
Two separate GPT-4o calls decide what to explore next:

1. **Link retrieval:** which linked node IDs are worth opening? Nodes already collected are filtered out.
2. **Footer search:** for footers that say "see paragraph x", produce a natural-language search query such as *"Return paragraph x to answer the query '…'"*.

If both lists are empty, the workflow goes to the Answering Agent.

### 4.7 Recursive Retrieval
- Moves the current nodes into `previous_nodes`.
- **Link targets** are looked up directly by node ID. An unknown ID is logged as a search failure.
- **Footer queries** run a keyword + BM25 search (`BM25Keyword`), then a pruning pass; empty results are logged as failures.
- The newly fetched nodes become `last_fetched_context_nodes`, `pass_count` increases, and the flow goes back to **Context Fetch** (clause expansion, link hints, footer parsing, supervisor, router). This loop is the "recursive" part: each pass can open the next layer of references (6.3 → 7.3 → 9.1).

The loop ends when the router finds nothing more to explore, or the supervisor says END, or LangGraph's recursion limit (150) is reached.

### 4.8 Answering Agent
GPT-4o receives every collected node (previous passes and the last pass), including their children, link hints and footer info, and writes the answer. It is instructed to cite sections or paragraphs but not to print raw node IDs or hashes.

---

## 5. Design decisions

- **Graph over flat chunks.** Structure (`contains`, `is_parent`, `follows`) restores clause integrity, and `links_to` makes cross-references traversable.
- **Geometry-based link recovery.** Hyperlinks are mapped to document elements by proximity, so they work with any layout parser that returns bounding boxes.
- **LLMs choose; the graph supplies.** For definitions and links, the model only selects *which* terms or nodes matter. Text is always read from the graph, which avoids hallucinated definitions and invented node IDs.
- **Hybrid retrieval.** BM25 catches exact legal wording and paragraph numbers ("Para 7.3"); vectors catch paraphrase.
- **Informative node IDs.** A content preview in the ID lets the LLM judge a link without opening it.
- **Prune early and often.** Pruning after each retrieval keeps the working set small and under the token budget.
- **Bounded recursion.** The supervisor, token budget and recursion limit stop the agent from wandering.

---

## 6. Configuration

| Setting | Default | Notes |
|---|---|---|
| Embedding model | `text-embedding-3-small` | Must be identical at indexing and query time. `text-embedding-3-large` (3072 dims) is recommended if you use AND mode in the hybrid retriever |
| LLM for agents | `gpt-4o-2024-08-06` | Pruning, definitions, footers, supervisor, router, answering, definition extraction |
| Initial retrieval | 1 vector + 7 BM25, mode OR | AND mode only works well with 3072-dim embeddings |
| Footer-driven search | 5 keyword + 3 BM25, mode OR | |
| Token budget | 32,000 (`o200k_base`) | Checked by the supervisor |
| Recursion limit | 150 | LangGraph |
| Link-matching height factor | 0.96 | Reducto vertical stretch correction |
| Definition pages | per document (e.g. pages 4-5) | Pages GPT-4o reads for terms |

---

## 7. LangGraph Agentic Workflow
![LangGraph_diagram](LangGraph/langgraph.svg)

---

## 8. Dependencies

`networkx` · `pymupdf` · `openai` · `pydantic` · `llama-index-core` · `llama-index-embeddings-openai` · `llama-index-llms-openai` · `llama-index-retrievers-bm25` · `PyStemmer` · `langgraph` · `langchain-core` · `tiktoken` · `pandas` · `numpy`

Inputs: the source PDF and its Reducto JSON output (`result → chunks → blocks`).
