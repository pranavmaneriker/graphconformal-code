import torch
import sys
sys.path.insert(1,'../')
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

fig, ax = plt.subplots(1, 2,figsize=(14,6.5))
plt.subplots_adjust(left=0.05, right=0.95, bottom = 0.1, top=0.8)

sns.set_theme(font_scale=1.4, style = "white", palette="tab10")
# plt.figure()
# sns.displot(df, x="score", hue="method", kind="kde", ax=ax[0])
sns.kdeplot(df, x="score", hue="method", ax=ax[0], legend=False)
ax[0].axvline(q_A, color=sns.color_palette()[0], linestyle="--", label=r"q_A")
ax[0].axvline(q_tA, color=sns.color_palette()[1], linestyle="--", label=r"q_{\Tilde{A}}")
ax[0].set_xlabel('Score', fontsize=14)
ax[0].set_ylabel('Density', fontsize=14)
ax[0].tick_params(labelsize=14)
ax[0].tick_params(labelsize=14)
#plt.legend()
# plt.savefig("aps_dist.png")

mask = torch.ones(probs.shape, dtype=torch.bool)
mask = mask.scatter_(1, labels.unsqueeze(1), False)
# breakpoint()


# Randomly sample both non-conformity scores: 
a_rand = torch.randint(probs.shape[1]-1,labels.shape)
ta_rand = torch.randint(probs.shape[1]-1,labels.shape)

a_rand[a_rand >= labels] += 1 
ta_rand[ta_rand >= labels] += 1  


nonlabel_scores_A =  s_A.gather(1, a_rand.unsqueeze(1)).squeeze().sort().values
nonlabel_scores_tA =  s_tA.gather(1, ta_rand.unsqueeze(1)).squeeze().sort().values


# nonlabel_scores_A = s_A[mask]
# nonlabel_scores_tA = s_tA[mask]

# breakpoint()
# nonlabel_scores_A = nonlabel_scores_A.sort().values
# nonlabel_scores_tA = nonlabel_scores_tA.sort().values

a_A = torch.argwhere(nonlabel_scores_A > q_A).min()/len(nonlabel_scores_A)
a_tA = torch.argwhere(nonlabel_scores_tA > q_tA).min()/len(nonlabel_scores_tA)

df = pd.DataFrame({
    "score": torch.concat([nonlabel_scores_A, nonlabel_scores_tA]).tolist(),
    "method": ["randomized"] * len(nonlabel_scores_A) + ["non-randomized"] * len(nonlabel_scores_tA),
    "sorted_index": torch.concat([torch.arange(len(nonlabel_scores_A))/len(nonlabel_scores_A), torch.arange(len(nonlabel_scores_tA))/len(nonlabel_scores_tA)]).tolist()
})


sns.lineplot(data=df, x="sorted_index", y="score", hue="method", ax=ax[1])
ax[1].axvline(a_A, color=sns.color_palette()[0], linestyle="--", label=r'$1 - \alpha^A_c$')
ax[1].axvline(a_tA, color=sns.color_palette()[1], linestyle="--", label=r'$1 - \alpha^{\widetilde{A}}_c$')
ax[1].axhline(q_A, color=sns.color_palette()[0], linestyle="-.", label=r'$q_A$')
ax[1].axhline(q_tA, color=sns.color_palette()[1], linestyle="-.", label=r'$q_{\widetilde{A}}$')
ax[1].set_xlabel('Sorted Index', fontsize=14)
ax[1].set_ylabel('Score', fontsize=14)
ax[1].legend()
ax[1].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)
sns.move_legend(ax[1], "upper left", bbox_to_anchor=(-0.63, 1.26), ncol=3)
fig.savefig("aps_exp_run.pdf")