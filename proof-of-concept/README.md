Proof-of-concept work for FEP-biased NNs. We have transfer-learned a structure-based neural network that predicts ligand-protein binding affinities by freezing the base model weights, appending several hidden fully connected layers and iteratively training the extended model.



Using the HIF2A dataset (42 ligands), we performed 15 replicates of randomly split 32/10/10 ligands (train/val/test), where train and val sets were trained on the FEP dG label and model performance was tested on testset EXP dG to see how translative the predictions are in situations where only FEP data is available.



See plot_pcc.ipynb for a (sloppy) notebook that contains code to plot the biasing progression. See the folder POC_WORK for all inputs/code/outputs. ML protocols were run on a system with 4x GTX1080s.

Results are computed as:
FEP (green): FEP+ MUE on whole ligand series
ML (blue): OnionNet MUE on whole ligands series
FEP-biased ML (blue): mean of MUEs (15 replicates) on held-out testset (i.e. vs EXP; see above), plotted as the roling global minimum.





![](<./POC_PLOT.png>)
