
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────

CATEGORIES = ['Food', 'Transport', 'Shopping', 'Entertainment', 'Bills', 'Income', 'Other']

# FIX 1: Weighted — Food & Bills dominate, not uniform
CATEGORY_WEIGHTS = [0.28, 0.18, 0.14, 0.08, 0.17, 0.10, 0.05]

DESC_MAP = {
    'Food': [
        # delivery apps
        'Swiggy', 'Zomato', 'Dunzo Grocery', 'Blinkit', 'Zepto',
        # restaurants & cafes
        'Restaurant', 'Cafe Coffee Day', 'Starbucks', 'Chai Point',
        'Barista', 'Third Wave Coffee', 'Haldirams', 'Saravana Bhavan',
        # fast food
        'Pizza Hut', 'Dominos', "McDonald's", 'KFC', 'Burger King',
        'Subway', 'Wow Momo', 'Biryani Blues', 'Faasos', 'Box8',
        # grocery & supermarket
        'Groceries', 'Big Bazaar', 'DMart', 'Reliance Fresh',
        'More Supermarket', "Spencer's", 'Nilgiris', "Nature's Basket",
        # bakery & sweet shops
        'Bakery', 'Monginis', 'Corner House', 'Naturals Ice Cream',
        'Baskin Robbins', 'Food Court',
    ],
    'Transport': [
        # ride hailing
        'Uber', 'Ola', 'Rapido', 'InDrive', 'Meru Cab', 'Blu-Smart',
        # fuel
        'Petrol', 'Diesel', 'HP Petrol Pump', 'Indian Oil', 'BPCL Fuel',
        # public transport
        'Metro Card Recharge', 'Bus Ticket', 'BMTC Bus Pass',
        'KSRTC Ticket', 'DTC Bus', 'BEST Bus',
        # inter-city
        'Train Ticket', 'IRCTC Booking', 'RedBus Ticket',
        'IntrCity SmartBus', 'Volvo Bus Ticket', 'Shatabdi Ticket',
        # local
        'Auto Rickshaw', 'E-Rickshaw', 'Parking Fee',
        'Toll Charge', 'FASTag Recharge', 'Cab Fare',
        # maintenance
        'Car Service', 'Tyre Puncture Repair', 'Vehicle Insurance',
        'Bike Service', 'Car Wash',
    ],
    'Shopping': [
        # e-commerce
        'Amazon', 'Flipkart', 'Meesho', 'Snapdeal', 'JioMart', 'Tata Cliq',
        # fashion
        'Myntra', 'Ajio', 'Nykaa Fashion', 'Lifestyle Store',
        'Westside', 'FabIndia', 'Max Fashion', 'Pantaloons', 'Zara', 'H&M',
        # electronics
        'Croma', 'Reliance Digital', 'Electronics Store',
        'Apple Store', 'Samsung Store', 'Vijay Sales', 'Poorvika',
        # beauty & personal care
        'Nykaa', 'Purplle', 'Health and Glow', 'The Body Shop', 'Mamaearth',
        # home & furniture
        'IKEA', 'Pepperfry', 'HomeTown', 'Wooden Street', 'Urban Ladder',
        # sports
        'Decathlon', 'Nike Store', 'Adidas Store', 'Puma Store',
        # books & stationery
        'Crossword', 'Landmark Books', 'Stationery Store', 'Amazon Books',
        # general
        'Shopping Mall', 'Clothing Store', 'Gift Shop', 'D2H Store',
    ],
    'Entertainment': [
        # OTT subscriptions
        'Netflix', 'Amazon Prime Video', 'Disney+ Hotstar',
        'SonyLIV', 'ZEE5', 'JioCinema', 'Apple TV+', 'Voot',
        # music & podcast
        'Spotify', 'YouTube Premium', 'Gaana', 'JioSaavn', 'Hungama Music',
        # movies & events
        'Movie Ticket', 'BookMyShow', 'INOX Cinema', 'PVR Cinema',
        'Carnival Cinemas', 'Cinepolis',
        # live events
        'Concert Ticket', 'Comedy Show', 'Theatre Ticket',
        'Stand-up Comedy', 'Sunburn Festival', 'Lollapalooza Ticket',
        # gaming
        'Steam Games', 'PlayStation Store', 'Google Play Games',
        'Mobile Game Purchase', 'Xbox Game Pass',
        # experiences
        'Amusement Park', 'Bowling Alley', 'Escape Room',
        'Go-Karting', 'Laser Tag', 'Paintball', 'Virtual Reality',
        # fitness-leisure
        'Gym Membership', 'Yoga Class', 'Swimming Pool', 'Cult.fit',
    ],
    'Bills': [
        # utilities
        'Electricity Bill', 'Water Bill', 'Gas Bill',
        'Piped Gas Bill', 'Municipal Tax', 'Property Tax',
        # telecom
        'Mobile Recharge', 'Postpaid Bill', 'Airtel Bill',
        'Jio Postpaid', 'Vi Recharge', 'BSNL Recharge', 'Vodafone Bill',
        # internet
        'Broadband Bill', 'Jio Fiber', 'Airtel Xstream',
        'ACT Fibernet', 'Hathway Internet', 'BSNL Broadband',
        # DTH & cable
        'DTH Recharge', 'Tata Play', 'Dish TV', 'Sun Direct', 'Airtel DTH',
        # insurance & EMI
        'Life Insurance Premium', 'Health Insurance Premium',
        'Vehicle Insurance EMI', 'Home Loan EMI',
        'Personal Loan EMI', 'Credit Card Bill', 'Car Loan EMI',
        # subscriptions
        'iCloud Storage', 'Google One', 'Microsoft 365',
        'Adobe Subscription', 'Antivirus Renewal', 'Dropbox',
        # society & housing
        'Society Maintenance', 'House Rent', 'Water Tank Charge',
        'Lift Maintenance', 'Security Charges',
    ],
    'Income': [
        # salary & employment
        'Salary Credit', 'Monthly Salary', 'Weekly Wages',
        'Bonus Credit', 'Incentive Payment', 'Arrears Credit',
        'Performance Bonus', 'Joining Bonus', 'Gratuity Credit',
        # freelance & consulting
        'Freelance Payment', 'Consulting Fee', 'Project Payment',
        'Contract Payment', 'Upwork Transfer', 'Fiverr Payout',
        'Toptal Payment', 'Freelancer.com Transfer',
        # investments
        'Interest Credit', 'FD Interest', 'RD Maturity',
        'Dividend Credit', 'Mutual Fund Redemption',
        'Stock Profit Transfer', 'PPF Interest', 'NPS Credit',
        # rental & passive
        'Rental Income', 'Airbnb Payout', 'Subletting Income',
        'PG Rent Received',
        # refunds & cashback
        'Refund Credit', 'GST Refund', 'Income Tax Refund',
        'Cashback Credit', 'Insurance Claim', 'Warranty Refund',
        'TDS Refund', 'Amazon Refund', 'Flipkart Refund',
        # other income
        'Scholarship Credit', 'Stipend', 'Pension Credit', 'Lottery Prize',
    ],
    'Other': [
        # cash & transfers
        'ATM Withdrawal', 'Cash Deposit', 'UPI Transfer',
        'NEFT Transfer', 'IMPS Transfer', 'RTGS Transfer', 'Cheque Deposit',
        # gifts & social
        'Gift Purchase', 'Wedding Gift', 'Birthday Gift',
        'Festival Shopping', 'Anniversary Gift',
        # charity
        'Donation', 'NGO Contribution', 'Temple Donation',
        'PM Relief Fund', 'CRY Donation', 'Goonj Donation',
        # medical
        'Medical Store', 'Apollo Pharmacy', 'Netmeds',
        '1mg Order', 'Doctor Consultation', 'Lab Test Fee',
        'Hospital Bill', 'Dental Clinic',
        # education
        'Course Fee', 'Udemy Course', 'Coursera', 'Tuition Fee',
        'School Fee', 'College Fee', 'Coaching Classes',
        # travel (one-off)
        'Hotel Booking', 'MakeMyTrip', 'Goibibo', 'OYO Rooms',
        'Flight Ticket', 'Cleartrip', 'Yatra Booking',
        # misc
        'Newspaper Subscription', 'Laundry', 'Salon',
        'Pet Care', 'Cobbler', 'Tailoring', 'Home Repair',
    ],
}

# FIX 2: Per-category lognormal (mean, sigma) -> realistic rupee ranges
AMOUNT_CONFIG = {
    'Food':          (4.5, 0.5),   # ~Rs 50-400
    'Transport':     (4.0, 0.5),   # ~Rs 30-300
    'Shopping':      (6.5, 0.8),   # ~Rs 300-8000
    'Entertainment': (5.2, 0.6),   # ~Rs 100-1500
    'Bills':         (6.8, 0.5),   # ~Rs 500-6000
    'Income':        (10.5, 0.4),  # ~Rs 20000-100000
    'Other':         (5.0, 1.0),   # wide range
}


# ── NumPy GAN ─────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def leaky(x, a=0.2):
    return np.where(x > 0, x, a * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def he_init(fan_in, fan_out):
    W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
    b = np.zeros(fan_out)
    return W, b

def bce(pred, target):
    eps = 1e-8
    return -np.mean(target * np.log(pred + eps) + (1 - target) * np.log(1 - pred + eps))

def bce_grad(pred, target):
    eps = 1e-8
    return (pred - target) / (pred * (1 - pred) + eps)


class Generator:
    """noise_dim -> 64 -> 128 -> 64 -> output_dim  (tanh output)"""
    def __init__(self, noise_dim, output_dim):
        self.W1, self.b1 = he_init(noise_dim, 64)
        self.W2, self.b2 = he_init(64, 128)
        self.W3, self.b3 = he_init(128, 64)
        self.W4, self.b4 = he_init(64, output_dim)

    def forward(self, z):
        self.z  = z
        self.h1 = relu(z        @ self.W1 + self.b1)
        self.h2 = relu(self.h1  @ self.W2 + self.b2)
        self.h3 = relu(self.h2  @ self.W3 + self.b3)
        self.out = np.tanh(self.h3 @ self.W4 + self.b4)
        return self.out

    def backward(self, d_out, lr):
        N = len(self.z)
        d = d_out * (1 - self.out ** 2)         # tanh grad
        gW4 = self.h3.T @ d / N;  gb4 = d.mean(0)
        d = (d @ self.W4.T) * (self.h3 > 0)     # relu grad
        gW3 = self.h2.T @ d / N;  gb3 = d.mean(0)
        d = (d @ self.W3.T) * (self.h2 > 0)
        gW2 = self.h1.T @ d / N;  gb2 = d.mean(0)
        d = (d @ self.W2.T) * (self.h1 > 0)
        gW1 = self.z.T  @ d / N;  gb1 = d.mean(0)
        self.W4 -= lr * gW4;  self.b4 -= lr * gb4
        self.W3 -= lr * gW3;  self.b3 -= lr * gb3
        self.W2 -= lr * gW2;  self.b2 -= lr * gb2
        self.W1 -= lr * gW1;  self.b1 -= lr * gb1


class Discriminator:
    """input_dim -> 128 -> 64 -> 32 -> 1  (sigmoid output)"""
    def __init__(self, input_dim):
        self.W1, self.b1 = he_init(input_dim, 128)
        self.W2, self.b2 = he_init(128, 64)
        self.W3, self.b3 = he_init(64, 32)
        self.W4, self.b4 = he_init(32, 1)

    def forward(self, x):
        self.x  = x
        a = 0.2
        self.h1 = leaky(x       @ self.W1 + self.b1, a)
        self.h2 = leaky(self.h1 @ self.W2 + self.b2, a)
        self.h3 = leaky(self.h2 @ self.W3 + self.b3, a)
        self.out = sigmoid(self.h3 @ self.W4 + self.b4)
        return self.out

    def backward(self, d_out, lr):
        N = len(self.x)
        a = 0.2
        d = d_out
        gW4 = self.h3.T @ d / N;  gb4 = d.mean(0)
        d = (d @ self.W4.T) * np.where(self.h3 > 0, 1, a)
        gW3 = self.h2.T @ d / N;  gb3 = d.mean(0)
        d = (d @ self.W3.T) * np.where(self.h2 > 0, 1, a)
        gW2 = self.h1.T @ d / N;  gb2 = d.mean(0)
        d = (d @ self.W2.T) * np.where(self.h1 > 0, 1, a)
        gW1 = self.x.T  @ d / N;  gb1 = d.mean(0)
        self.W4 -= lr * gW4;  self.b4 -= lr * gb4
        self.W3 -= lr * gW3;  self.b3 -= lr * gb3
        self.W2 -= lr * gW2;  self.b2 -= lr * gb2
        self.W1 -= lr * gW1;  self.b1 -= lr * gb1

    def grad_wrt_input(self):
        """Gradient of D output w.r.t. its input (used to train G)."""
        a = 0.2
        d = np.ones((len(self.x), 1))
        d = (d @ self.W4.T) * np.where(self.h3 > 0, 1, a)
        d = (d @ self.W3.T) * np.where(self.h2 > 0, 1, a)
        d = (d @ self.W2.T) * np.where(self.h1 > 0, 1, a)
        return d @ self.W1.T


# ── Build seed data ───────────────────────────────────────────────

def build_seed_data(n=3000):
    rows = []
    for _ in range(n):
        idx = np.random.choice(len(CATEGORIES), p=CATEGORY_WEIGHTS)
        cat = CATEGORIES[idx]
        mean, sigma = AMOUNT_CONFIG[cat]
        amt = np.random.lognormal(mean=mean, sigma=sigma)
        # encode: normalised amount + normalised category index
        rows.append([amt, idx / (len(CATEGORIES) - 1)])
    return np.array(rows, dtype=np.float32)


# ── Train GAN ─────────────────────────────────────────────────────

def train_gan(seed_data, noise_dim=16, epochs=600, batch=64, lr=0.001):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    real   = scaler.fit_transform(seed_data)
    n, dim = real.shape

    G = Generator(noise_dim, dim)
    D = Discriminator(dim)

    ones  = np.ones((batch, 1))
    zeros = np.zeros((batch, 1))

    for ep in range(epochs):
        # ── Discriminator ──
        idx  = np.random.randint(0, n, batch)
        Xr   = real[idx]
        z    = np.random.randn(batch, noise_dim)
        Xf   = G.forward(z)

        D.forward(Xr);  D.backward(bce_grad(D.out, ones),  lr)
        D.forward(Xf);  D.backward(bce_grad(D.out, zeros), lr)
        d_loss = bce(D.forward(Xr), ones) + bce(D.forward(Xf), zeros)

        # ── Generator ──
        z  = np.random.randn(batch, noise_dim)
        Xf = G.forward(z)
        D.forward(Xf)
        g_loss = bce(D.out, ones)
        # gradient: fool D -> target ones
        d_grad_in = bce_grad(D.out, ones)
        # chain through D layers to get grad w.r.t G output
        a = 0.2
        d = d_grad_in
        d = (d @ D.W4.T) * np.where(D.h3 > 0, 1, a)
        d = (d @ D.W3.T) * np.where(D.h2 > 0, 1, a)
        d = (d @ D.W2.T) * np.where(D.h1 > 0, 1, a)
        d = d @ D.W1.T
        G.backward(d, lr * 0.5)

        if (ep + 1) % 150 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs}  D_loss={d_loss:.4f}  G_loss={g_loss:.4f}")

    return G, scaler, noise_dim


# ── Generate final dataset ────────────────────────────────────────

def generate_synthetic_transactions(n_samples=1000, noise_dim=16, epochs=600):
    print("Step 1: Building seed data...")
    seed = build_seed_data(n=3000)

    print("Step 2: Training GAN...")
    G, scaler, noise_dim = train_gan(seed, noise_dim=noise_dim, epochs=epochs)

    print(f"Step 3: Generating {n_samples} transactions from trained GAN...")
    z        = np.random.randn(n_samples, noise_dim)
    gan_out  = G.forward(z)
    gan_real = scaler.inverse_transform(gan_out)

    rows = []
    for i in range(n_samples):
        # FIX 1: Weighted category
        cat_idx = np.random.choice(len(CATEGORIES), p=CATEGORY_WEIGHTS)
        cat     = CATEGORIES[cat_idx]
        desc    = np.random.choice(DESC_MAP[cat])

        # FIX 2: GAN amount clipped to per-category range
        mean, sigma = AMOUNT_CONFIG[cat]
        lo  = np.exp(mean - 2.5 * sigma)
        hi  = np.exp(mean + 2.5 * sigma)
        amt = round(float(np.clip(abs(gan_real[i, 0]), lo, hi)), 2)

        # FIX 4: Full year + realistic time of day per category
        day_offset = np.random.randint(0, 365)
        # Each category has a typical active hour range
        hour_ranges = {
            'Food':          (8,  22),   # breakfast to late dinner
            'Transport':     (7,  21),   # morning commute to evening
            'Shopping':      (10, 21),   # malls/online open hours
            'Entertainment': (17, 23),   # evenings and nights
            'Bills':         (9,  18),   # office hours
            'Income':        (9,  11),   # salary credited early morning
            'Other':         (8,  22),   # any time
        }
        lo_h, hi_h = hour_ranges[cat]
        hour   = np.random.randint(lo_h, hi_h + 1)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        date = (pd.to_datetime('2024-01-01')
                + pd.to_timedelta(day_offset, unit='D')
                + pd.to_timedelta(hour * 3600 + minute * 60 + second, unit='s'))

        rows.append([date, desc, amt, cat])

    df = pd.DataFrame(rows, columns=['Date', 'Description', 'Amount', 'Category'])
    return df.sort_values('Date').reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_synthetic_transactions(n_samples=1000)
    df.to_csv("synthetic_transactions.csv", index=False)

    print("\n[OK] Saved synthetic_transactions.csv")
    print(f"     Shape : {df.shape}")
    print(f"     Dates : {df['Date'].min().date()}  to  {df['Date'].max().date()}")

    print("\nCategory distribution (target vs actual):")
    dist = df['Category'].value_counts()
    target = dict(zip(CATEGORIES, CATEGORY_WEIGHTS))
    for cat, cnt in dist.items():
        bar = '#' * (cnt // 8)
        pct = cnt / len(df) * 100
        tgt = target.get(cat, 0) * 100
        print(f"  {cat:<15} {cnt:>4} ({pct:4.1f}%)  target={tgt:.0f}%  {bar}")

    print("\nAmount stats per category (Rs):")
    stats = df.groupby('Category')['Amount'].agg(['min', 'mean', 'max']).round(2)
    print(stats.to_string())

    print("\nSample rows:")
    print(df.sample(8, random_state=1).sort_values('Date').to_string(index=False))
