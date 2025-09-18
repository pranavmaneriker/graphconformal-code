# graphconformal-code

### Environment export from conda

```
conda env export | grep -v "name" | grep -v "prefix" > environment.yml
```

### Directory Structure

**graph_conformal:** Main library for graph conformal prediction

**configs:** YAML configs for experiments. The folder includes the best model configurations (base GNN and CFGNN) based on hyperparameter tuning.

**scripts:** SLURM scripts for job execution

**analysis:** Notebook for plot generation

The remaining Python files are used to run hyperparameter tuning and conformal prediction.


## Acknowledgements
The authors acknowledge support from National Science Foundation (NSF) grant #2112471 (AI-EDGE) and a grant from Cisco Research (US202581249). Any opinions and findings are those of the author(s) and do not necessarily reflect the views of the granting agencies.
