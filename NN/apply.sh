 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/NN/run.py b/NN/run.py
index fe7fcde079c9b6de2b146cdd5e940516b43daf40..68f448cc7a1e238c341ddfd1016cf27cf4e9ba10 100644
--- a/NN/run.py
+++ b/NN/run.py
@@ -29,50 +29,80 @@ num_repeats = 5
 data_seed = 123 # it is used for random_state of splitting data into source and train sets
 # changing it creates different source and train sets.
 # Since the number of data is low changing it can largely affect the results
 model_seed = 123 # it is used for random_state of models
 def set_model_seed(model_seed):
     torch.manual_seed(model_seed)
     np.random.seed(model_seed)
 
 learning_rate = 0.05
 # The learnign rate used in ANN
 #hidden_size1 = 10
 hidden_size1 = 15
 hidden_size2 = 10
 
 # the number of neurons in hidden layers
 
 # https://alexlenail.me/NN-SVG/
 # use the site above to draw the following network
 #
 hidden_sizes = [15, 10, 3 ]
 # nn.ReLU(), nn.Tanh(), nn.Identity()
 
 list_hidden_sizes = [[10], [15, 10, 3], [8, 4], [15, 5]]
 normalization_type = "z_score"
 
+
+def parse_multi_select(answer, options, allow_all=True):
+    """Parse space-separated indexes and return selected values."""
+    if not answer:
+        return None if allow_all else []
+
+    answer = answer.strip().lower()
+    if allow_all and answer == "all":
+        return None
+
+    selected_values = []
+    for token in answer.split():
+        if not token.isdigit():
+            raise ValueError(f"Invalid input '{token}'. Please use numeric indexes.")
+        index = int(token)
+        if index < 0 or index >= len(options):
+            raise ValueError(
+                f"Invalid selection '{index}'. Valid range is 0 to {len(options) - 1}."
+            )
+        selected_values.append(options[index])
+
+    if len(set(selected_values)) != len(selected_values):
+        raise ValueError("Duplicate selections are not allowed.")
+    return selected_values
+
+
+def ask_with_default(prompt, default):
+    answer = input(f"{prompt} [{default}]:").strip()
+    return answer if answer else str(default)
+
 # Define the normalization function
 def normalize(data, normalization_type):
     if normalization_type == 'z_score':
         return (data - data.mean()) / data.std()
     elif normalization_type == 'minmax':
         return (data - data.min()) / (data.max() - data.min())
     else:
         raise ValueError("Unsupported normalization type. Choose 'z_score' or 'minmax'.")
 
 # Function to apply model on data and generate predictions
 # Return predictions, MSE and R-Squared
 import torch.nn.init as init
 
 def fit_model(model, X_train, X_test, y_train, y_test, 
         num_epochs, display_steps=False, run=0):
 
     set_model_seed(model_seed + run)
     scaler = StandardScaler()
     X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
     X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
     y_train_values = y_train.values if hasattr(y_train, "values") else y_train
     y_test_values = y_test.values if hasattr(y_test, "values") else y_test
     y_train = torch.tensor(y_train_values, dtype=torch.float32)
     y_test = torch.tensor(y_test_values, dtype=torch.float32)
     if y_train.ndim == 1:
@@ -288,192 +318,202 @@ def forward_feature_selection(model_class, data, inputs, output, num_epochs, hid
 
     print('\n\n')
     print('=============== Final Features ==================')
     print('Selected Features: ')
     print(candidates)
     print('Final R2: ' + str(best_r2))
     print('Elminated Features: ')
     print(set(data.columns).difference(candidates))
 
     table = pd.DataFrame(data=rows)
     return table
 
 
 # Repeats an fit_model to get average of results
 def repeat_fit_model(model_class, num_repeats, 
         num_epochs, hidden_sizes, 
         display_steps=False, features=None):
     X_train, X_test, y_train, y_test = read_prep_data(features)
     r2_list = []
     mse_list = []
     max_r2 = 0
     max_run = 0
     best_preds = None
     input_size = X_train.shape[1]
     output_size = y_train.shape[1] if hasattr(y_train, "shape") and len(y_train.shape) > 1 else 1
-    try:
-        model = model_class(input_size, hidden_sizes, output_size=output_size)
-    except TypeError:
-        model = model_class(input_size, hidden_sizes)
-    if len(hidden_sizes) == len(model.hidden_layers):
-        for i in range(num_repeats):
-            predictions, mse, r2, model = fit_model(model, 
-                    X_train, X_test, y_train, y_test, num_epochs, 
-                    display_steps=display_steps, run=i)
-            if r2 is None:
-                continue
-
-            if r2 > max_r2:
-                max_r2 = r2
-                max_run = i
-                best_preds = predictions
-
-            r2_list.append(r2*100)
-            mse_list.append(mse)
-
-        if display_steps:
-            print(r2_list)
+    for i in range(num_repeats):
+        # Recreate model on each run so repeats are independent.
+        try:
+            model = model_class(input_size, hidden_sizes, output_size=output_size)
+        except TypeError:
+            model = model_class(input_size, hidden_sizes)
+
+        if len(hidden_sizes) != len(model.hidden_layers):
+            continue
+
+        predictions, mse, r2, model = fit_model(model, 
+                X_train, X_test, y_train, y_test, num_epochs, 
+                display_steps=display_steps, run=i)
+        if r2 is None:
+            continue
+
+        if r2 > max_r2:
+            max_r2 = r2
+            max_run = i
+            best_preds = predictions
+
+        r2_list.append(r2*100)
+        mse_list.append(mse)
+    if display_steps:
+        print(r2_list)
 
     mean_r2 = np.mean(r2_list) if r2_list else None
     mean_mse = np.mean(mse_list) if mse_list else None
     std_r2 = np.std(r2_list) if r2_list else None
     std_mse = np.std(mse_list) if mse_list else None
 
     return mean_r2, std_r2, mean_mse, best_preds, max_r2*100, r2_list, max_run
 
 ############################### Start of Program ###################
 # Sync prep_data with current dataset (NN/data.csv by default).
 dataset_path="data.csv"
 data = pd.read_csv(dataset_path)
 output_features, input_features = resolve_data_columns(
     data,
     output_feature=None,
     input_features=None
 )
 # User input for selecting the model and number of epochs
 answer = input("\n".join([str(i) + ")" + name for i,name in enumerate(input_features)]) \
         + "\nSelect one or several input_features (separated with space) [all]:")
 
-if not answer:
-    answer = "all"
-if answer.lower() == "all":
-    selected_input_features = None
-else:
-    indexes = answer.split()
-    selected_input_features = []
-    for ind in indexes:
-        feature_index = int(ind)
-        if feature_index > len(input_features):
-            print("Invalid feature selection. Please enter all or 1 to ", len(input_features) - 1)
-            exit()
-        selected_input_features.append(input_features[feature_index])
+try:
+    selected_input_features = parse_multi_select(answer, input_features, allow_all=True)
+except ValueError as err:
+    print(f"Invalid input feature selection: {err}")
+    exit()
 
 answer = input(
     "\n".join([str(i) + ")" + name for i, name in enumerate(data.columns)])
     + "\nSelect one or several output features (separated with space) [last column]:"
 )
-if not answer:
-    selected_output_features = [data.columns[-1]]
-else:
-    indexes = answer.split()
-    selected_output_features = []
-    for ind in indexes:
-        feature_index = int(ind)
-        if feature_index >= len(data.columns):
-            print("Invalid output feature selection.")
-            exit()
-        selected_output_features.append(data.columns[feature_index])
+try:
+    selected_output_features = (
+        [data.columns[-1]]
+        if not answer
+        else parse_multi_select(answer, data.columns.tolist(), allow_all=False)
+    )
+except ValueError as err:
+    print(f"Invalid output feature selection: {err}")
+    exit()
 
 sync_prep_data_with_dataset(
     dataset_path=dataset_path,
     prep_folder="prep_data",
     input_features=selected_input_features,
     output_feature=selected_output_features
 )
 
 # Load data from prep_data after schema sync.
 X_train, X_test, y_train, y_test = read_prep_data(inputs=None, prep_folder="prep_data")
 
 # After loading, get the column names from X_train
 inputs = X_train.columns.tolist()
 outputs = y_train.columns.tolist()
 output = outputs
 
 data = X_train
 print("inputs:", inputs)
 print("outputs:", outputs)
 ans = input("Are these inputs and outputs for files in prep_data folder correct?(y/n):")
+if ans.strip().lower() not in ("y", "yes"):
+    print("Please re-run and choose your preferred input/output features.")
+    exit()
 
 # Dynamically collect all model classes from the module
 models = [
     member for name, member in inspect.getmembers(models, inspect.isclass)
     if issubclass(member, models.nn.Module) and member.__module__ == models.__name__
 ]
 
 model_names=[model.__name__ for model in models]
 # User input for selecting the model and number of epochs
 answer = input("\n".join([str(i) + ")" + name for i,name in enumerate(model_names)]) \
         + "\nSelect one or several models (separated with space) [all]:")
 
 if not answer:
     answer = "all"
 
-if answer.lower() == "all":
-    selected_models = range(len(models)) # choose the index of all models
+try:
+    selected_model_names = parse_multi_select(answer, model_names, allow_all=True)
+except ValueError as err:
+    print(f"Invalid model selection: {err}")
+    exit()
+
+if selected_model_names is None:
+    selected_models = list(range(len(models)))
 else:
-    indexes = answer.split()
-    selected_models = []
-    for ind in indexes:
-        model_index = int(ind)
-        if model_index > len(models):
-            print("Invalid model selection. Please enter all or 1 to ", len(models) - 1)
-            exit()
-        selected_models.append(model_index)
+    selected_models = [model_names.index(name) for name in selected_model_names]
 
 print("Selected Models:", [model_names[i] for i in selected_models])
 best_model_index = selected_models[0]
 max_model_index = best_model_index
 
-answer = input(f"Enter the number of epochs [{list_epochs}] (0 to skip training):")
+answer = ask_with_default("Enter the number of epochs separated by spaces (or 0 to skip training)", " ".join([str(e) for e in list_epochs]))
 if answer != "0":
-    if answer: 
-       list_epochs = [int(a) for a in answer.split() if a.isnumeric()]
+    if answer:
+       list_epochs = [int(a) for a in answer.split() if a.isnumeric() and int(a) > 0]
+    if not list_epochs:
+       print("No valid epoch values were provided.")
+       exit()
     
     print(list_epochs)
 
-    answer = input(f"Enter the hidden sizes [{list_hidden_sizes}]:")
+    answer = ask_with_default(
+        "Enter hidden sizes (groups split by '#', e.g. '10 5 # 15 10 3')",
+        " # ".join([" ".join([str(v) for v in hs]) for hs in list_hidden_sizes]),
+    )
     if answer: 
        list_hidden_sizes = []
        hs = answer.split("#")
        for ans in hs:
           ans = ans.strip()
-          h = [int(a) for a in ans.split()]
-          list_hidden_sizes.append(h)
-
-    answer = input(f"Enter the number of repeating predictions [{num_repeats}]:")
-    if answer: 
+          if not ans:
+              continue
+          h = [int(a) for a in ans.split() if a.isdigit() and int(a) > 0]
+          if h:
+              list_hidden_sizes.append(h)
+    if not list_hidden_sizes:
+       print("No valid hidden size values were provided.")
+       exit()
+
+    answer = ask_with_default("Enter the number of repeating predictions", num_repeats)
+    if answer:
        num_repeats = int(answer)
+       if num_repeats < 1:
+           print("Repeat count must be at least 1.")
+           exit()
 
 
     best_mean_r2 = -1000
     best_mse = -1000
     best_hidden_sizes = []
     best_r2 = -1000
     best_run = 0
     max_epochs = -1
     max_hidden_sizes = []
     best_epochs = -1
     results = []
     model_best_predictions = {}
     # for all models
     for model_index in selected_models:
         for num_epochs in list_epochs:
             for hidden_sizes in list_hidden_sizes:
                 # Instantiate the selected model
                 model_class = models[model_index]
                 model_name = model_names[model_index]
                 # Apply model on data for 3 times and get predictions, mse and r2
                 mean_r2, std_r2, mean_mse, model_best_preds, max_r2, r2_list, max_run = repeat_fit_model(
                         model_class,
                         num_repeats, num_epochs, hidden_sizes, display_steps=True)
 
                 # Keep best seed to generate the same predictions later
 
EOF
)