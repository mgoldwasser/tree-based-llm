# Task 07: True Residual Boosting with Feature Pass-Through

**Status:** Completed

## What was done
BoostedForest stages now receive `[original_input, current_output]` concatenated, so each stage can see both raw features and what previous stages predicted — enabling it to learn residual corrections.

## Key changes
- Stage input dimension: `input_dim + output_dim` (was just `input_dim`)
- `stage_input = torch.cat([x, output], dim=-1)` before each stage
- Each stage's BatchedTreeForest accepts the wider input

## Impact
Stages now have access to the residual information. Tree routing can naturally learn to focus on output dimensions where correction is needed.
