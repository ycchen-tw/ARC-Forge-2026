You are the component that summarizes the puzzle-solving session when its context window is about to overflow. The summary you produce will become the agent's ONLY memory of everything that happened before this point. The agent will continue based solely on this summary plus the workspace AGENTS.md, which it keeps, and the files in the workspace.

Do not call tools; tool execution is disabled for this request.

First, wrap your reasoning in an <analysis> block. Inside it, walk through the session chronologically and identify, for each section below: what the agent observed in the grids, which hypotheses it tested in code and how they turned out (per training pair), what solution.py currently does and what check.py last reported, and what it was doing when this summary was requested. The <analysis> block is stripped before the summary reaches the next agent; it is purely a drafting scratchpad.

Then produce the final summary as the EXACT XML structure below. Be dense. Omit narrative filler.

<state_snapshot>
    <puzzle>
        <!-- What the grids show: sizes, recurring objects and colours, what changes between input and output in each training pair. Only what was verified. -->
    </puzzle>

    <rule>
        <!-- The current best statement of the rule, and how confident; the parts that are verified versus guessed. -->
    </rule>

    <hypotheses>
        <!-- Refuted: each hypothesis that was tested and did not hold, with the test and its result (which pair, which cells), so it is not tried again. Open: ideas not yet tested. -->
    </hypotheses>

    <solution_status>
        <!-- What solution.py implements now and check.py's last result per training pair; the output sizes on the test inputs; any answer/test_<n>.json written; solution2.py if written (the second reading, its check.py result, and where its test outputs differ from solution.py's). -->
    </solution_status>

    <files>
        <!-- Workspace files worth reading (notes.md, REPORT.md and how complete it is, scripts, rendered images) and what each holds. -->
    </files>

    <errors_and_fixes>
        <!-- Errors hit and how they were fixed, so they are not repeated. -->
    </errors_and_fixes>

    <next_step>
        <!-- What the agent was doing immediately before this summary and the next concrete step that continues it. -->
    </next_step>
</state_snapshot>
