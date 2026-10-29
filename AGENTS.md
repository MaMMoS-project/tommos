# Agent Rules

## Rule 0: Global Pre-flight Checklist
At the start of every message, you MUST explicitly output a markdown block titled **`[Rule Compliance Checklist]`**. In this checklist, you must list every rule relevant to your upcoming response and, if applicable, cite the exact tool call ID used to satisfy it before proceeding with your reasoning or actions.

## Rule 1: NO TOOL EXECUTION WITHOUT APPROVAL
**CRITICAL:** Do NOT use the `replace_file_content`, `multi_replace_file_content`, or `write_to_file` tools under ANY circumstances until the user explicitly writes the word **[Approved]**. Always describe in detail what you plan to do. Once you propose a plan, you MUST immediately stop calling tools and end your turn. 

## Rule 1.1: Mandatory Compliance Check-in
Before invoking any file-editing tool, you must explicitly output the exact phrase: *"I am verifying that the user has explicitly approved this specific code change."* If you cannot find the explicit approval in the immediate chat history, you must abort the tool execution.

## Rule 2: Relevant Code Changes and Improvements
You are prohibited from using file-editing tools on any file that the user did not explicitly mention in their prompt. If you identify an optional improvement in another file, you MUST output the exact header **`# [UNREQUESTED IMPROVEMENT]`** and you are prohibited from generating a plan until the user responds with `[Approved]`.

## Rule 3: Interaction Sequence for Fixes
If a `run_command` tool call returns an exit code other than 0, you MUST immediately halt execution and output a markdown block titled **`[ERROR DIAGNOSIS]`** containing your analysis. You are strictly prohibited from invoking further tools (like `run_command` or file editors) to attempt a fix until you receive the word `[Approved]`.

## Rule 4: Mechanical Syntax Check Before Execution
Before invoking the `run_command` tool to execute a Python script (e.g., `python3 loop.py`), you MUST explicitly output the phrase: **'I verify that a previous `run_command` execution of `python3 -m py_compile [filename]` returned exit code 0.'** If you cannot reference a successful compile step in the chat context, you are prohibited from executing the script.

## Rule 5: No Commits Without Quoted Proof
Before invoking the `run_command` tool with `git commit`, you must actively fetch and output the exact terminal or log output proving the code succeeded. 
* **For local tests (pytest/python):** You must quote the `run_command` stdout showing passing tests.
* **For Slurm jobs:** You must use the `grep_search` or `view_file` tool to read the generated `.log` file and quote the success indicator.

You must explicitly output the exact phrase: *"I verify success via the following log output: [insert exact log quote]"*. If you cannot quote the success output from a tool call in your immediate context, you must abort the commit.

## Rule 6: No Assumptions Without Confirmation
When using the `replace_file_content` or `multi_replace_file_content` tool, if your replacement chunk deletes more than 5 consecutive lines of existing code, or removes existing comments/docstrings, you MUST explicitly output: **'Warning: This plan deletes existing code.'** You are prohibited from executing the edit unless the user's prompt specifically instructed you to delete it.

## Rule 7: Mechanical Side Effect Search
Before generating a plan to modify any Python function or class, you MUST invoke the `grep_search` tool to search for all occurrences of that function/class name across the entire `src/` directory. You must explicitly list all files that import or use the target before proposing your changes.

## Rule 8: Stick to the Approved Plan
If you encounter an error (e.g. from `run_command`) that prevents you from completing an already approved plan, you MUST output the exact phrase: **'Plan execution blocked by error.'** You are strictly prohibited from generating alternative file edits until you have presented a new plan and received a new `[Approved]` keyword.

## Rule 9: Mechanical Evidence Verification (Anti-Hallucination)
When citing log outputs or file contents to the user, you MUST enclose the exact text in a markdown block prefixed with **`[EXTRACTED EVIDENCE]`**. You are strictly prohibited from generating an `[EXTRACTED EVIDENCE]` block unless the text is a verbatim substring of output returned by a `run_command`, `grep_search`, or `view_file` tool call in the current or immediate prior turn.

## Rule 10: Citation and Verifiability
Before stating any factual claim regarding third-party libraries, physics, or system behaviors, you MUST explicitly output `[Source: URL/File]` immediately following the claim. You are strictly prohibited from generating this citation unless you have actively executed the `search_web`, `read_url_content`, or `view_file` tool to read that exact source during the current session.

## Rule 11: Mechanical Link Verification
You are prohibited from generating markdown hyperlinks `[text](http...)` to external sites unless you have explicitly invoked the `read_url_content` tool (or `run_command` with `curl -I`) on that exact URL in the current or previous turn. Before generating any external hyperlink, you MUST explicitly output a block titled `[LINK VERIFICATION]` containing the exact HTTP 200 OK output from your tool call. If you cannot quote the successful status code, you must drop the link.

## Rule 12: Resolve Contradictions
Before proposing a diagnosis for an error, you MUST output a markdown table titled **'Hypothesis Check'** with two columns: [Current Hypothesis] and [Previous Hypothesis]. If they differ, you MUST output the exact phrase: **'I was wrong previously.'** before proposing the new fix.

## Rule 13: The "Dead End" Admission Rule (Anti-Pivot)
If you attempt to mechanically verify a claim, fact, or hypothesis and the verification fails (e.g., `grep_search` finds no results, a URL returns 404, or a log file lacks the expected output), you MUST immediately halt execution and output a markdown block titled **`[VERIFICATION FAILURE]`** detailing exactly what failed. 

You are strictly **prohibited** from proposing an alternative hypothesis, generating a new claim, or pivoting to a new solution in the same response. You must explicitly state: *"I cannot verify my claim and must stop."* and immediately end your turn to await user guidance.

## Rule 13: Never change stable tests
Tests in the directory `tests/stable` are supposed to guarantee that the CLI always stay valid and that physical behavior is reproduced. Never change or delete them.



## LaTeX Paper Writing Rules
- **The Preservation Rule**: When using `replace_file_content` on a `.tex` file, your `ReplacementContent` MUST be equal to or greater in length (character count) than your `TargetContent`. If you attempt to delete more text than you add, you MUST output: **'Warning: This edit reduces the document length'** and you are prohibited from executing the edit without explicit `[Approved]` permission.
- **The Precision Rule**: Immediately before drafting any new LaTeX text containing math blocks, you MUST output a list titled **'Variable Definitions'**. For every single symbol used in the draft, you must list its definition. If a symbol in your draft is missing from this list, you are prohibited from writing the file.
- **The Fact-Checking Rule**: Before invoking `write_to_file` on a `.tex` file, if the text contains a factual claim about physics or history, you MUST output a citation to a specific file or URL you have read using `view_file` or `read_url_content` in the current session. If no such tool was called, you are prohibited from writing the text.
- **The Post-Writing Fact-Check Rule**: Immediately after invoking `write_to_file` or a replacement tool on a `.tex` file, you MUST generate a markdown table titled **'Fact-Check Matrix'** with 3 columns: [Claim], [Source Citation], and [Verification Status]. You cannot proceed to any other task or end the session until this table is generated to verify the newly added text against primary sources.
- **The Formal Tone Rule**: Avoid colloquialisms, conversational language, and dramatic or "fancy" phrasing. Always write in a strictly formal, objective, and academic tone appropriate for a scientific publication.
- **The Consistency Rule**: Before calling `write_to_file` or a replacement tool on a `.tex` file to add new mathematical formulas, you MUST explicitly output a markdown table titled **'Consistency Check'** that maps your new variables to the output of a prior `grep_search` or `view_file` tool call. You are prohibited from writing the file until you prove you have computationally searched for existing definitions.
- **Plotting**:
  - **Legibility**: When writing or modifying Python plotting code (e.g. `matplotlib`), you MUST explicitly include `plt.rcParams.update({'font.size': 14})` (or larger).
  - **Distinguishability**: Differentiate data curves using multiple visual cues simultaneously (e.g., combine distinct colors with varying line styles or marker symbols).
  - **Titles**: You are prohibited from generating plot titles using `plt.title()`; all descriptions must go in captions.

## Coding Style & Types
- **Documentation**: All functions must use **Google-style docstrings** without type hints in the docstrings. Each parameter must be documented separately.
- **Type Safety**: Use explicit **Python type hints** for all parameters and return types. Use `jnp.ndarray` (or `Array` alias) for JAX arrays and `np.ndarray` for CPU/IO data.
