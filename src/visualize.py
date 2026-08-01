from load_data import load_data
import matplotlib.pyplot as plt

df = load_data()

# رسم هیستوگرام همه ستون‌ها
df.hist(figsize=(14, 10))

plt.tight_layout()

# ذخیره تصویر
plt.savefig("images/histograms.png")

plt.show()