# Berkelium generation evaluation

correct = grounded & correct; wrong = confidently wrong; ungrnd = right answer without evidence; invalid = unparsable actions. Higher correct, lower wrong is better.

Scoring policy: every episode the MODEL caused - unparsable output, illegal or invalid actions, budget exhaustion, ungrounded or wrong declarations - is scored. An episode in which the DETERMINISTIC CORE raised an exception (environment, solver, judge) is an infra_error: it is excluded from every rate, counted in the 'infra' column, listed by task id below, and retried on the next run. Saved episodes are re-validated on every resume by replaying their recorded actions through the current environment; any whose judged outcome would change are re-run, so one table never mixes results from two versions of the environment.

## claim

| gen | attempted | scored | infra | correct | wrong | ungrnd | invalid | return |
|---|---|---|---|---|---|---|---|---|
| base | 10 | 10 | 0 | 0.00 | 0.00 | 0.00 | 1.00 | -1.00 |
| m1 | 10 | 10 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.60 |
| *cheapest_sufficient* | 10 | 10 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.60 |
| *overconfident* | 10 | 10 | 0 | 0.00 | 0.00 | 1.00 | 0.00 | -1.00 |

## model

| gen | attempted | scored | infra | correct | wrong | ungrnd | invalid | return |
|---|---|---|---|---|---|---|---|---|
| base | 10 | 10 | 0 | 0.00 | 1.00 | 0.00 | 1.00 | -2.00 |
| m1 | 10 | 10 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.88 |
| *constructor* | 10 | 10 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.88 |
| *greedy_retrieval* | 10 | 10 | 0 | 0.60 | 0.40 | 0.00 | 0.00 | -0.29 |

## model_seen

| gen | attempted | scored | infra | correct | wrong | ungrnd | invalid | return |
|---|---|---|---|---|---|---|---|---|
| base | 10 | 10 | 0 | 0.00 | 1.00 | 0.00 | 1.00 | -2.00 |
| m1 | 10 | 10 | 0 | 0.80 | 0.20 | 0.00 | 0.00 | 0.32 |
| *constructor* | 10 | 10 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.90 |
| *greedy_retrieval* | 10 | 10 | 0 | 0.30 | 0.70 | 0.00 | 0.00 | -1.21 |

