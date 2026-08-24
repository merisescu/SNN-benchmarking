#%%
import scripts.api.setup as setup
import scripts.api.dataset as dataset

test = dataset.SSC_loader()

test.setup_speech_commands(all_keywords=False)
data, labels = test.train_dataset[5]
spikes, labels = test.training_loader()
print(spikes)
print(labels)

# %%
