# Baselines

The task is the same as for the protomaml part. We have multiple datasets with
different classes and we try to to learn a meta learner. For the baseline
we can take multiple approaches:

**Meta learning baseline**

These baselines could be used to see if the meta learning process is beneficial. These do not
use episodic learning.

- Finetune a bert model with a K-way softmax, where K is the total number of classes
over all the datasets. (should we do a softmax over all classes or individual softmax for each task) This resulting bert model has learned a good embedding space. 
  During few shot evaluation we classify a query to its nearest support neightbour in 
  the embedding space. [Eleni Triantafillou et al. (2020)]
- The same as the above but during few shot evaluation we train a new output linear layer
on the support. With this new output network we can do inference. [Eleni Triantafillou et al. (2020)]


**Implementation**

As its implemented now there is a shared MLP on top of bert. On top of that each task
has it's own small mlp. During few shot learning this last MLP is retrained.

It currently gets a batch from a dataset, updates the weights, gets a batch from another
dataset etc... It might be better to combine all datasets in a single update?
  
**Todos**
- [ ] Inference function for knn
- [ ] Inference function for finetune

