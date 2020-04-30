from bn.train import train, ModelType

train(model_type=ModelType.FF, num_epochs=2)
train(model_type=ModelType.BN, num_epochs=2)