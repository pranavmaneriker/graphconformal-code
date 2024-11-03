# graphconformal-code

### Environment export from conda

```
conda env export | grep -v "name" | grep -v "prefix" > environment.yml
```

### Directory Structure

**graph_conformal:** Main library for graph conformal prediction

**configs:** YAML configs for experiments. The folder includes the best model (base GNN and CFGNN) configurations based on the hyperparameter tuning.

**scripts:** SLURM scripts for job execution

**analysis:** Notebook for plot generation

Remaining python files are used to run hyperparameter tuning and conformal prediction.
