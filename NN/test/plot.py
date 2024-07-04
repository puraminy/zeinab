import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# داده‌های نمونه به صورت DataFrame
data = {
    'x': [1, 2, 3, 4, 5],
    'y1': [2, 3, 5, 7, 11],
    'y2': [1, 4, 6, 8, 10]
}

df = pd.DataFrame(data)

# رسم نمودار با استفاده از seaborn
sns.lineplot(x='x', y='y1', data=df, label='y1')
sns.lineplot(x='x', y='y2', data=df, label='y2')

# افزودن عنوان و برچسب‌ها
plt.title('Comparing y1 and y2 over x')
plt.xlabel('x')
plt.ylabel('Values')

# نمایش افسانه
plt.legend()

# نمایش نمودار
plt.show()
