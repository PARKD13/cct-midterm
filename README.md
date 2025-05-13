This project implements a Bayesian Cultural Consensus using PyMC to estimate the competent of informants and the consensus answers to a set of binary questions about plant knowledge.

Our plant_knowledge.csv dataset consisted of 10 informants and 20 yes or no questions which were modeled in a Bernouli distribution based on each resepctive informants latents competence and the predicted consensus answer to each question.

We assigned a Uniform(0.5, 1.0) prior to each informant’s competence, based on the theoretical assumption from CCT that competence must be at least 0.5 to reflect a understanding that is greater than just 50/50 chance. 

For each consensus answer (Zj), we used a Bernoulli(0.5) prior, expressing no initial bias toward either 0 or 1. We defined the likelihood of each response Xij with a Bernoulli distribution. Posterior inference was performed with 4 chains of 2000 samples each, including 1000 tuning steps.

The Convergence diagnostics from our summary confirmed that all R-hat values were near 1.0, and posterior distributions were smooth, indicating good convergence.

Informant 6 had the highest posterior mean competence, while Informant 3 had the lowest. The Consensus answers were estimated by rounding the posterior mean probabilities for each Zj​. Compared to a naive majority vote, the CCT-based consensus differed on several questions. These differences highlight how the CCT model discounts responses from less competent informants, offering a more robust cultural signal than simple aggregation.

In ambiguous cases (e.g., prior selection for DiD_iDi​), I opted for the Uniform(0.5, 1.0) prior as it adheres to CCT's theoretical assumptions and avoids overly constraining the model. Future extensions could explore hierarchical priors for competence or account for guessing behavior explicitly.

