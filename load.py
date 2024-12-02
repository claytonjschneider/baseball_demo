import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from models import LinearRegressionModel

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from pdb import set_trace

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

TEST_SIZE = 0.1
BATCH_SIZE = 64
SEED = 42

def nick_wan_dataset():
    data = {}
    for fname in ['train', 'test']:
        data[fname] = pd.read_csv(os.path.join(os.pardir, 'nwds', f"{fname}.csv"))
        print("Loaded dataset with shape: ",
              data[fname].shape)

    return data['train']  #, data['test']

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)

def clean(train):

    """
    Test columns:
    ['uid', 'pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z', 'game_type', 'is_lhp', 'is_lhb', 'balls', 'strikes', 'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'is_top', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate', 'release_extension', 'release_pos_y', 'pitch_number', 'pitch_name', 'spin_axis', 'spray_angle', 'bat_speed', 'swing_length']
    Evaluation columns:
    ['outcome', 'outcome_code']
    """
    
    print("Cleaning:")
    print("="*10)
    train.drop('outcome', axis=1, inplace=True)
    train.drop([
        'pitch_type',
        'game_type'],
        axis=1, inplace=True)
    # test.drop([
    #     'pitch_type',
    #     'game_type'],
    #     axis=1, inplace=True)

    # for now, one-hot encode only non-numeric data (pitch type)
    # this is sparse, adding 15 features
    pitch_types = ['4-Seam Fastball', 'Curveball', 'Screwball', 'Knuckle Curve', 'Cutter', 'Knuckleball', 'Slurve', 'Eephus', 'Changeup', 'Sinker', 'Split-Finger', 'Sweeper', 'Slider', 'Forkball', 'Other']

    train = encode_and_bind(train, 'pitch_name')
    # test  = encode_and_bind(test,  'pitch_name')

    train.drop('pitch_name', axis=1, inplace=True)
    # test.drop('pitch_name', axis=1, inplace=True)

    outcomes = train['outcome_code']
    outcomes = pd.get_dummies(outcomes)

    outcomes = outcomes.astype(float)
    outcomes = torch.tensor(outcomes.values).double()
    # outcomes = outcomes.unsqueeze(len(outcomes.shape)).double()

    train.drop('outcome_code', axis=1, inplace=True)
    train = train.apply(lambda x: x.astype(float), axis=1)
    train = torch.tensor(train.values).double()

    # test = test.apply(lambda x: x.astype(float), axis=1)
    # test = torch.tensor(test.values).double()

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(train)),
        outcomes,
        stratify=outcomes,
        test_size=TEST_SIZE,
        random_state=SEED)

    # generate subset based on indices
    train_split = Subset(train, train_indices)
    test_split  = Subset(train, test_indices)

    train_split_outcomes = Subset(outcomes, train_indices)
    test_split_outcomes  = Subset(outcomes, test_indices)

    train          = torch.stack([t for t in train_split])
    train_outcomes = torch.stack([t for t in train_split_outcomes])
    test           = torch.stack([t for t in test_split])
    test_outcomes  = torch.stack([t for t in test_split_outcomes])

    train_dataset = TensorDataset(train, train_outcomes)
    test_dataset  = TensorDataset(test,    test_outcomes)

    return train_dataset, test_dataset

def prompt_for_ready(prompt="train model"):
    print(f"Ready to {prompt}? y/N")
    user_in = input()
    while user_in not in ["\n", "y", "N"]:
        print(f"You input {user_in}")
        print("Valid inputs are y/N (just press enter for N)")
        print(f"Ready to {prompt}? y/N")
        user_in = input()
    
    if user_in in ["N", '\n']:
        print("Raising debugger shell. Enter 'c' when you are ready to step in.")
        set_trace()

def learn_linear_regression(train):
    
    num_features = len(train[0][0])

    # create batches
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    model = LinearRegressionModel(num_features=num_features, num_outcomes=5)

    criterion = torch.nn.MSELoss(size_average = False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters consumed: {total_params}")

    return model

def evaluate(model, test):

    data_loader  = DataLoader(test,  batch_size=BATCH_SIZE)
    
    criterion = torch.nn.MSELoss(size_average = False)

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        model.eval()

        for data, target in data_loader:
            y_pred = model(data)
            test_loss = criterion(y_pred, target)
            
            _, predicted_class = torch.max(y_pred, 1)
            _, real_class      = torch.max(target, 1)

            total_loss += test_loss.item()
            correct_samples += predicted_class.eq(real_class).sum()

    accuracy = 100 * correct_samples / total_samples
    print('Test Loss:', total_loss)
    print('Test Accuracy:', accuracy)

# Using the special variable 
# __name__
if __name__=="__main__":

    # training, outcomes
    train, test = clean(nick_wan_dataset())

    model = learn_linear_regression(train)

    evaluate(model, test)