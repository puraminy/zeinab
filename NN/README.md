## اجرای برنامه

برای اجرای برنامه به فولدر برو و دستور

```
python model.py
```
اگر خطا داد که Module XXX not found اون ماژول رو نصب کن که الان ماژول tabulate نیاز به نصب داره که با دستور زیر هر جا که اجرا کنی نصب می شه... 




```
pip install tabulate
```



## مدلها و نتایج
به توضیحات داخل برنامه توجه کن تا با خود مدلها و روند برنامه که خیلی قسمتهاش توضیح داده شده آشنا بشی.

برای خود مقاله می توی این توضیحات رو بنویسی و با چت جی پی تی به انگیلیسی ترجمه کنی:


ما انواع شبکه های عصبی را استفاده کردیم که دو شبکه عصبی خطی و چهار شبکه عصبی غیر خطی است. شبکه ها شامل یک یا دو لایه مخفی هستند و در شبکه های غیر خطی از توابع غیر خطی tanh و ReLU استفاده کردیم. در لایه اول از ۱۰ نرون و در لایه دوم از ۵ نرون استفاده شده است. 

مدلها را با ۳۰۰ ایپاک و نرخ یادگیری 0.05 اجرا کردیم. از آنجا که مدلهای عصبی با تغییر داده و همینطور وزن اولیه نتایج متفاوتی ایجاد می کنند ما آزمایش ها یا اجراها را ۵ بار تکرار کرده و نتایج میانگین را برای مدلها گزارش کردیم. در جدول .... نتایج R2 برای این مدلها نشان داده شده است. همینطور انحراف معیار ناشی از تکرار آزماش‌ها مشخش شده است. بهترین نتیجه میانگین برابر با ۹۶ درصد بدست آمد و بهترین نتجیه تک اجرا برابر با ۹۸.۶ درصد بدس آمد

(دقت کن که تغییر پارامترها مثل نرخ یادگیری و تعداد لایه و تعداد نرون در لایه می تونه نتایج را عوض کن و شاید بهبود بده ولی تنظیمات فعلی هم نتایج خوبی داشتند)

طبق این نتایج بهترین نتیجه R2 میانگین برای مدل با دو لایه پنهان و با استفاده از تابع غیرخطی ReLU بدست آمد که با عنوان Relu2HiddenLayer مشخص شده است. 

### مدلهای خطی
در مدلهای خطی یک تابع خطی بر روی داده ها فیت (برازش) شده است. اما دقت نتایج مدلهای خطی بالا نیست. نتایج مدلهای غیرخطی به مراتب بهتر از مدل‌های خطی است  و این نکته نشان می دهد که رابطه HTC و سایر پارامترها رابطه‌ای غیرخطی است. همینطور در مدل‌های خطی تعداد لایه تاثیرگذار نیست و در واقع این نتیجه بهترین نتیجه برای فیت کردن یک تابع خطی رو داده‌هاست.

### مدلهای غیرخطی
در مدل‌های غیرخطی دو لایه مخفی عملکرد بهتری داشته است که نشان‌دهنده پیچیدگی رابطه غیرخطی است. همینطور مدل‌های غیرخطی با دولایه انحراف معیار کمتر و ثبات بیشتری در نتایج را نشان داده اند. (در مورد توابع کمی بنویس و شاید تفاوتهایی که دارند و غیره)

نمودار پیش‌بینی برای داده‌های آزمایشی با استفاده از بهترین مدل بدست آمده در شکل فلان....

## انتخاب ویژگی موثر
پس از تعیین بهترین مدل در این قسمت سعی داریم بهترین ویژگی‌ها برای پیش‌بینی htc را بدست آوریم و آزمایش‌های زیر با استفاده از بهترین مدل که همان مدل دولایه غیر خطی با تابع relu بود انجام شده اند.


ما برای تعیین ویژگی های موثر در پشبینی htc از روشهای انتخاب ویژگی  backward feature elimination و forward feature selection استفاده کردیم. 


در روش حذف ویژگی عقب رو ما ابتدا نتیجه را با حفظ همه ویژگیها بدست می آوریم و سپس با حذف هر کدام از ویژگیها نتایج را ثبت می کنیم و به این ترتیب اگر نتایج جدید بالاتر از نتایج قبل شد ویژگی مربوطه را حذف می کنیم. نتایج در جدول  .... نشان داده شده است. مشخص است که بهترین نتیجه با حفظ کلیه ویژگیها بدست می آید و حذف هر کدام از ویژگی‌ها نتیجه را کاهش می دهد. هرچند تاثیر حذف پارامترها متفاوت است و طبق نتایج ویژگیهای flow_rate و ..... تاثیر بیشتری دارند. (توضیحات بیشتر)

نتایج روش انتخاب ویژگی رو به جلو در جدول ... آمده است.
در روش رو به جلو ابتدا آزمایش را با هر کدام از ویژگیها به تنهایی بدست می آوریم و سپس بهترین ویژگی را نتخاب کرده و ترکیب آن با هرکدام از ویژگیهای باقیمانده را بدست می آوریم. چنانچه نتایج بهتر شد سپس بهترین ترکیب دو تایی را انتخاب کرده و سپس با افزودن هر کدام از ویژگیهای باقیمانده ترکیب سه تایی ساخته و نتایج ره بدست می آوریم. البته این روش کلیه ترکیبات ممکن را آزمایش نمی کند و طبق نتایج بهترین نتیجه با ترکیب heat_flux, flow_rate, Kfluid بدست آمده است که جز پارامترهای موثر بوده اند و افزودن هر ویژگی دیگر به تنهایی این نتیجه را بهبود نبخشیده اما طبق روش قبل بهترین نتیجه مربوط به استفاده از تمامی ویژگی‌هاست.

(یک توضیحاتی که چرا اینها پارامترهای موثری هستند.... )


## فایلهای لتکس مربوط به نتایج

نتایج اجرا علاوه بر نمایش روی صفحه در فایلهای داخل پوشه tables به شکل جدول در قالب لتکس قرار دارن که می تونی کپی و داخل مقاله پیست کنی...  البته روشی هم هست که به خود همین فایل ارجاع بدی تا خودکار از فایل بخونه ولی باشه برای بعدها... شکلها نیز به تریب دقت در فولد images ذخیر می شن که می تونی وارد لتکس کنی .. با کپی تصویر و سپس پیست روی خود ویرایشگر در texstudio هم اینکار انجام می شه... 




## Industrial refinery variable logic and target-leakage prevention

The training workflow now follows industrial refinery timing.  Model inputs are no longer selected from every non-target column.  They are limited to variables that are known before downstream quality is measured, plus variables that plant operators can directly adjust.

### Variable groups

1. `EARLY_VARIABLES`
   - Variables available early in the refinery process.
   - Current columns: `sheet_name`, `shift_name`, `raw_sugar_color`, `raw_syrup_brix`, `raw_syrup_color`.

2. `CONTROL_VARIABLES`
   - Operator-adjustable process variables.
   - Current columns: `lime_alkalinity`, `co2_percent`, `carbonated_alkalinity`, `carbonated_pH`.
   - These represent controllable items such as lime/alkalinity, CO2, and pH control.

3. `TARGET_VARIABLES`
   - Future or downstream quality outputs.
   - Current columns: `filtercake_moisture`, `filtercake_sugar`, `sweetwater_brix`, `sulphited_pH`, `sulphited_brix`, `sulphited_color`, `standard_liquor_pH`, `standard_liquor_brix`, `standard_liquor_color`, `white_total_points`.

### Leakage rule

Only `EARLY_VARIABLES + CONTROL_VARIABLES` are allowed in `X_train` and `X_test`.  `TARGET_VARIABLES` and the selected output columns are blocked as model inputs, including during "all" input selection, automatic feature selection, saved-run reuse, and temporal feature engineering.  This prevents target leakage from future quality measurements into the model.

## Sequence-aware industrial learning

The refinery workflow can now convert safe early/control inputs into sequence-aware process features without using future target variables as inputs.

### Added feature families

1. **Lag-window features**
   - For selected sequential columns, the data-preparation step can generate `__lag_1`, `__lag_2`, and `__lag_3` columns by default.
   - These features expose the recent refinery history to tabular ANN and sklearn models, so the prediction can react to delayed effects from previous batches, shifts, or rows.

2. **Time-sequence process features**
   - Difference, ratio, acceleration, and normalized-change features describe how refinery variables move from one observation to the next.
   - Ordered process groups also describe stage-to-stage movement between related process measurements, for example a downstream control value relative to an upstream value.

3. **Rolling process dynamics features**
   - Rolling mean, standard deviation, min, max, range, and slope features summarize short-window process stability and drift.
   - These features help the model detect noisy operation, persistent trends, and abrupt transitions rather than treating every row as an independent static sample.

### Why this improves refinery prediction

Refinery quality is not only controlled by the current value of pH, brix, alkalinity, CO2, or color.  It is also affected by recent operating history, delayed residence-time effects, stage-to-stage movement, and whether the process is stable or drifting.  Lag-window, sequence, and rolling dynamics features therefore give ordinary feed-forward and tree models some of the context normally captured by sequence models, while preserving the existing leakage-prevention rule that targets and downstream quality outputs must not be used as inputs.

## Prescriptive industrial AI recommendation engine

The project now includes a prescriptive recommendation layer in `recommendation_engine.py`.  The predictive model still learns future sugar-quality behavior, but the new `recommend_operating_conditions()` function uses that trained model to search actionable refinery set-points and recommend the best feasible controls.

### Inputs

`recommend_operating_conditions()` receives:

* the current refinery conditions for the same input columns used during training;
* a trained model, either a scikit-learn-style model with `predict()` or a PyTorch model with the input/target scalers attached by `run.py`;
* the trained input and output feature names;
* optional historical input/target data to estimate realistic operating windows and quality-risk thresholds.

### Controllable variables searched

By default the engine searches the requested industrial operator control levers:

* `lime_milk_baume`
* `lime_alkalinity`
* `co2_percent`
* `carbonated_pH`
* `sulphited_pH`
* `sulphited_brix`
* `standard_liquor_pH`
* `standard_liquor_brix`

The wider refinery control list still includes `carbonated_alkalinity`, but that variable is only optimized when explicitly requested.

### Industrial constraints

The search never changes raw/early process conditions.  It only changes approved control variables from `CONTROL_VARIABLES`, and it rejects any target/downstream quality leakage in the model inputs.  Each control is searched inside a realistic engineering envelope.  If historical data is supplied, the engine tightens the envelope to the observed 5th-to-95th percentile range so recommendations stay near proven plant operation.

### Objective logic

For every candidate set-point combination, the engine:

1. keeps all current non-controllable conditions fixed;
2. substitutes one candidate combination for the available requested operator set-points;
3. calls the trained model to predict future sugar quality/color;
4. scores the candidate by minimizing predicted future color/quality output;
5. classifies predicted sugar quality as `LOW`, `MEDIUM`, or `HIGH` industrial risk using historical quality thresholds (`LOW` at or below P50, `MEDIUM` between P50 and P80, and `HIGH` above P80 when history is available);
6. adds explicit MEDIUM/HIGH risk penalties so recommendations favor safer predicted sugar quality, not only a lower raw objective score;
7. adds a small movement and range-edge penalty so the recommendation avoids unnecessary set-point jumps and unrealistic boundary operation;
8. returns the feasible candidate with the lowest objective score.

The recommendation result now includes `risk_prediction`, `current_risk_prediction`, per-target risk drivers, and clear `operator_warnings`.  When `run.py` trains and saves a PyTorch checkpoint, it prints the recommended risk level and operator warnings, writes the full recommendation to `tables/recommended-operating-conditions.json`, and creates the operator-facing workbook at `reports/operator_report.xlsx`.

### Industrial operator demo output

After the best PyTorch model is saved, `run.py` now prints a professional operator-demo section for plant demonstrations.  The report follows the live refinery workflow:

1. the shift operator enters the current refinery conditions;
2. the AI predicts future sugar quality at the current conditions;
3. the AI recommends optimal controllable variables such as lime milk baumé, lime alkalinity, CO2 percentage, carbonated pH, sulphited pH/brix, and standard-liquor pH/brix;
4. the AI prints the expected future quality after applying the recommended set-points;
5. the AI prints the industrial risk level, risk drivers, and operator advisory messages.

The demo is formatted as readable shift-log tables so that engineers and operators can compare current values, recommended set-points, expected quality improvement, safe search ranges, and the final `LOW`, `MEDIUM`, or `HIGH` risk level during an industrial presentation.

## Sequence-model migration note

Before replacing the feedforward ANN with LSTM or GRU, see
[`sequence_model_evaluation.md`](sequence_model_evaluation.md). The current
recommendation is to keep the feedforward/tabular approach as the production
candidate, add leakage-safe temporal features where justified, and test a small
GRU/LSTM only as a controlled benchmark after grouped time-series validation is
available.

