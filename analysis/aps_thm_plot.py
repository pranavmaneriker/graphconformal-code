import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], ".."))
# hack to import graph_conformal here
import torch
from graph_conformal.config import PrimitiveScoreConfig
from graph_conformal.scores import APSScore
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

probs = torch.load("aps_thm_outputs/probs.pt")
labels = torch.load("aps_thm_outputs/labels.pt")

orig_config = PrimitiveScoreConfig(use_aps_epsilon=False)
orig_score = APSScore(orig_config)

our_config = PrimitiveScoreConfig(use_aps_epsilon=True)
our_score = APSScore(our_config)

s_tA = orig_score.compute(probs)
s_A = our_score.compute(probs)

label_scores_A =  s_A.gather(1, labels.unsqueeze(1)).squeeze()
label_scores_tA =  s_tA.gather(1, labels.unsqueeze(1)).squeeze()

df = pd.DataFrame({
    "score": torch.concat([label_scores_A, label_scores_tA]).tolist(),
    "method": ["randomized"] * len(label_scores_A) + ["non-randomized"] * len(label_scores_tA) 
})
alpha = 0.1
q_tA = orig_score.compute_quantile(label_scores_tA, alpha)
q_A = our_score.compute_quantile(label_scores_A, alpha)

plt.figure()
sns.displot(df, x="score", hue="method", kind="kde")
plt.axvline(q_A, color=sns.color_palette()[0], linestyle="--", label=r"q_A")
plt.axvline(q_tA, color=sns.color_palette()[1], linestyle="--", label=r"q_{\Tilde{A}}")
#plt.legend()
plt.savefig("figures/aps_dist.pdf")

mask = torch.ones(probs.shape, dtype=torch.bool)
mask = mask.scatter_(1, labels.unsqueeze(1), False)
nonlabel_scores_A = s_A[mask]
nonlabel_scores_tA = s_tA[mask]

nonlabel_scores_A = nonlabel_scores_A.sort().values
nonlabel_scores_tA = nonlabel_scores_tA.sort().values

a_A = torch.argwhere(nonlabel_scores_A > q_A).min()/len(nonlabel_scores_A)
a_tA = torch.argwhere(nonlabel_scores_tA > q_tA).min()/len(nonlabel_scores_tA)

df = pd.DataFrame({
    "score": torch.concat([nonlabel_scores_A, nonlabel_scores_tA]).tolist(),
    "method": ["randomized"] * len(nonlabel_scores_A) + ["non-randomized"] * len(nonlabel_scores_tA),
    "sorted_index": torch.concat([torch.arange(len(nonlabel_scores_A))/len(nonlabel_scores_A), torch.arange(len(nonlabel_scores_tA))/len(nonlabel_scores_tA)]).tolist()
})
plt.figure()
sns.lineplot(data=df, x="sorted_index", y="score", hue="method")
plt.axvline(a_A, color=sns.color_palette()[0], linestyle="--", label=r'$1 - \alpha^A_c$')
plt.axvline(a_tA, color=sns.color_palette()[1], linestyle="--", label=r'$1 - \alpha^{\widetilde{A}}_c$')
plt.axhline(q_A, color=sns.color_palette()[0], linestyle="-.", label=r'$q_A$')
plt.axhline(q_tA, color=sns.color_palette()[1], linestyle="-.", label=r'$q_{\widetilde{A}}$')
plt.legend()
plt.savefig("figures/aps_sorted.pdf")

