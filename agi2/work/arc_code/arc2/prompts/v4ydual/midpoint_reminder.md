Checkpoint: this conversation has grown long (`cat budget.txt` shows how much of your output budget is used).

If check.py does not yet pass every training pair, stop extending the current approach for a moment. Write in notes.md which facts in the training pairs your current rule fails to explain, then consider at least one substantially different reading of the puzzle -- what the objects are, what is copied and what is computed, what decides the value of each output cell -- and test it in code before you go back to refining.

If every training pair already passes, carry on: check transform's output on every test input against the rule, then stop.
