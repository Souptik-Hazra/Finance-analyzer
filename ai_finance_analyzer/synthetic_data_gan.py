import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(noise_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

def generate_synthetic_transactions(n_samples=1000, noise_dim=8, epochs=300):
    categories=['Food', 'Transport', 'Shopping', 'Entertainment', 'Bills', 'Income', 'Other']
    desc_map={
        'Food': ['Swiggy', 'Restaurant', 'Groceries', 'Cafe Coffee Day', 'Pizza Hut', 'Dominos', 'Big Bazaar', 'Food Court'],
        'Transport': ['Uber', 'Petrol', 'Ola', 'Metro', 'Bus Ticket', 'Train Ticket', 'Cab Fare'],
        'Shopping': ['Amazon', 'Shopping Mall', 'Flipkart', 'Myntra', 'Lifestyle', 'Electronics Store', 'Clothing Store'],
        'Entertainment': ['Netflix', 'Movie', 'BookMyShow', 'Concert', 'Amusement Park', 'Spotify', 'YouTube Premium'],
        'Bills': ['Electricity Bill', 'Water Bill', 'Internet Bill', 'Mobile Recharge', 'Gas Bill', 'DTH Recharge'],
        'Income': ['Salary', 'Freelance Payment', 'Bonus', 'Interest Credit', 'Refund'],
        'Other': ['Miscellaneous', 'Gift', 'Donation', 'Cashback', 'Lottery']
    }
    le=LabelEncoder()
    le.fit(categories)
    np.random.seed(42)
    synth_rows=[]
    for _ in range(n_samples):
        cat=np.random.choice(categories)
        desc=np.random.choice(desc_map[cat])
        amt=np.random.lognormal(mean=3, sigma=1.1)
        if np.random.rand() < 0.05:
            amt=np.random.uniform(0.01, 1)
        elif np.random.rand() < 0.05:
            amt=np.random.uniform(1e5, 1e6)
        if np.random.rand() < 0.01:
            amt=-amt
        day=np.random.randint(1, 32)
        date=pd.to_datetime('2024-01-01')+pd.to_timedelta(day-1, unit='D')
        synth_rows.append([date, desc, round(amt, 2), cat])
    synth=pd.DataFrame(synth_rows, columns=['Date', 'Description', 'Amount', 'Category'])
    return synth[['Date', 'Description', 'Amount', 'Category']]

if __name__ == "__main__":
    df=generate_synthetic_transactions(n_samples=1000)
    df.to_csv("synthetic_transactions.csv", index=False)
    print("Synthetic dataset saved as synthetic_transactions.csv")
