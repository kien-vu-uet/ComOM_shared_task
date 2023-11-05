import torch.optim as optim


def Baseline_Optim(model, optimizer_parameters):
    return optim.Adam([{'params': model.encoder.parameters(), 'lr': 1e-5},
                       {'params': model.sent_linear.parameters(), 'lr': 2e-5},
                       {'params': model.W.parameters(), 'lr': 0.001},
                       {'params': model.decoder.parameters(), 'lr': 0.005}], betas=(0.9, 0.99))


def LSTMModel_Optim(model, optimizer_parameters):
    return optim.Adam([{'params': model.encoder.parameters(), 'lr': 0.0001},
                       {'params': model.sent_linear.parameters(), 'lr': 2e-5},
                       {'params': model.W.parameters(), 'lr': 0.001},
                       {'params': model.decoder.parameters(), 'lr': 0.01}], betas=(0.9, 0.99))


def Logistic_Optim(model, optimizer_parameters):
    return optim.Adam([{'params': model.fc.parameters(), 'lr': 0.0005}], betas=(0.9, 0.99))

def MLP_Optim(model, optimizer_parameters):
    return optim.Adam([{'params': model.mlp.parameters(), 'lr': 1e-3}], betas=(0.9, 0.99))

def Transformer_cls_Optim(model, optimizer_parameters):
    return optim.Adam([{'params': model.attentions.parameters(), 'lr': 0.0005},
                       {'params': model.classifier.parameters(), 'lr': 0.0005}], betas=(0.9, 0.99))
