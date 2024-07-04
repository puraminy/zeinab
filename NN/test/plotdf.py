import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# خواندن داده‌ها از فایل CSV
df = pd.read_csv('../results.csv')

# نمایش داده‌های خوانده شده
print(df)

# رسم نمودار با استفاده از seaborn
sns.lineplot(x='HTC', y='Predictions', data=df, label='y1')
# sns.lineplot(x='HTC', y='y2', data=df, label='y2')

# افزودن عنوان و برچسب‌ها
plt.title('Comparing y1 and y2 over x')
plt.xlabel('x')
plt.ylabel('Values')

# نمایش افسانه
plt.legend()

# نمایش نمودار
plt.show()
