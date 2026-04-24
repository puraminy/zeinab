
diff --git a/NN/run.py b/NN/run.py
index d7eebcf4e547413f3c673c54e133672871a03b8c..1a57da8387859ed31ee2458d39cac2e824fd2bdd 100644
--- a/NN/run.py
+++ b/NN/run.py
@@ -31,76 +31,79 @@ data_seed = 123 # it is used for random_state of splitting data into source and
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
 
 # Define the normalization function
 def normalize(data, normalization_type):
+    eps = 1e-8
     if normalization_type == 'z_score':
-        return (data - data.mean()) / data.std()
+        std = data.std()
+        return (data - data.mean()) / (std + eps)
     elif normalization_type == 'minmax':
-        return (data - data.min()) / (data.max() - data.min())
+        data_range = data.max() - data.min()
+        return (data - data.min()) / (data_range + eps)
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
     y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
     y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
 
    # Normalize inputs and targets to zero mean and unity standard deviation
-    X_train_normalized = normalize(X_train, normalization_type)
-    X_test_normalized = normalize(X_test, normalization_type)
-    y_train_normalized = normalize(y_train, normalization_type)
-    y_test_normalized = normalize(y_test, normalization_type)
+    X_train_normalized = torch.nan_to_num(normalize(X_train, normalization_type), nan=0.0, posinf=0.0, neginf=0.0)
+    X_test_normalized = torch.nan_to_num(normalize(X_test, normalization_type), nan=0.0, posinf=0.0, neginf=0.0)
+    y_train_normalized = torch.nan_to_num(normalize(y_train, normalization_type), nan=0.0, posinf=0.0, neginf=0.0)
+    y_test_normalized = torch.nan_to_num(normalize(y_test, normalization_type), nan=0.0, posinf=0.0, neginf=0.0)
 
 
     # Initialize weights
     def weights_init(m):
         if isinstance(m, nn.Linear):
             init.kaiming_uniform_(m.weight.data)
             if m.bias is not None:
                 init.constant_(m.bias.data, 0)
 
     model.apply(weights_init)
 
     criterion = nn.MSELoss()
     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
     for epoch in range(num_epochs):
         model.train()
         optimizer.zero_grad()
         outputs = model(X_train_normalized)
         
         # Check for NaN in outputs
         if torch.isnan(outputs).any():
             print(f"NaN detected in outputs at epoch {epoch + 1}")
             return None, None, None, model
 
         loss = criterion(outputs, y_train_normalized)
@@ -435,112 +438,125 @@ if answer != "0":
                     best_run = max_run
 
                 if mean_r2 > best_mean_r2:
                     best_mean_r2 = mean_r2
                     best_mse = mean_mse
                     best_model_index = model_index
                     best_epochs = num_epochs
                     best_hidden_sizes = hidden_sizes
                 
                 total_nodes = sum(hidden_sizes)
 
                 result = {
                         "model":model_name, 
                         "R2": round(mean_r2,1), 
                         "MSE": round(mean_mse,2),
                         "R2 std": round(std_r2, 1),
                         "R2 List": [round(x, 1) for x in r2_list],
                         "hidden sizes": hidden_sizes,
                         "total hs": total_nodes,
                         "epochs": num_epochs,
                         }
                 results.append(result)
 
     # Creata a Table for results
     results_table = pd.DataFrame(data=results)
-    # Sort methods by R2
-    results_table = results_table.sort_values(by = "R2", ascending=False)
+    if results_table.empty:
+        print("No valid training runs were produced. Skipping result sorting/export.")
+    elif "R2" not in results_table.columns:
+        print("No R2 column found in results. Skipping result sorting/export.")
+    else:
+        # Sort methods by R2
+        results_table = results_table.sort_values(by = "R2", ascending=False)
     latex_table = results_table.copy()
     # Create and save latex code for table
-    latex_table["R2"] = latex_table.apply(lambda row: f"{row['R2']} ± {row['R2 std']}", axis=1)
-    latex_table = latex_table.drop(columns=["R2 List","R2 std"])
-    results_table_latex = generate_latex_table(latex_table, 
-            caption="Results of different models", label="models")
-    with open(os.path.join("tables", "results.tex"), 'w', encoding='utf-8') as f:
-        print(results_table_latex, file=f)
+    if not results_table.empty and "R2" in results_table.columns:
+        latex_table["R2"] = latex_table.apply(lambda row: f"{row['R2']} ± {row['R2 std']}", axis=1)
+        latex_table = latex_table.drop(columns=["R2 List","R2 std"])
+        results_table_latex = generate_latex_table(latex_table, 
+                caption="Results of different models", label="models")
+        with open(os.path.join("tables", "results.tex"), 'w', encoding='utf-8') as f:
+            print(results_table_latex, file=f)
 
     # Plot the performance of models across different parameters
-    print("Generating plots ...")
-    plot_model_performance(results_table)
+    if not results_table.empty and "R2" in results_table.columns:
+        print("Generating plots ...")
+        plot_model_performance(results_table)
 
     best_model = models[best_model_index]
     best_model_name = model_names[best_model_index]
     # Show results
     max_model_name = model_names[max_model_index]
 
     print("============ Results for models =========================")
     print(results_table)
     print("========================== Best Mean Model ===============================")
     print("Best Mean R-Squred:", best_mean_r2)
     print("Best model with better mean R-Squred:", best_model_name) 
     print("Best Hidden sizes:", best_hidden_sizes) 
     print("Best epochs:", best_epochs) 
     print("=========================== Max Model ================================")
     print("Best model with better max R-Squred:", max_model_name) 
     print("Best Hidden sizes:", max_hidden_sizes) 
     print("Best epochs:", max_epochs) 
     print("Max R-Squred:", best_r2)
  
-    results_table.to_csv("exp.csv")
+    if not results_table.empty and "R2" in results_table.columns:
+        results_table.to_csv("exp.csv")
 
     X_train, X_test, y_train, y_test = read_prep_data()
 
     # Show and save the plot for best results
-    best_predictions = model_best_predictions[max_model_name] 
+    if max_model_name not in model_best_predictions:
+        print("No valid predictions are available to plot.")
+    else:
+        best_predictions = model_best_predictions[max_model_name] 
     title = "Prediction of " + output + " with " + max_model_name + " epochs:" + str(max_epochs)
     file_name = f"R2-{best_r2:.2f}-" + max_model_name + "-" + output + ".png"
 
-    print("\n\n")
-    print("Plot was saved in plots folder")
-    answer = input("Do you want to see them? [y]:") 
-    if answer == "y" or answer == "yes":
-        plot_results(best_predictions, y_test, title, file_name, show_plot=True)
-    else:
-        plot_results(best_predictions, y_test, title, file_name, show_plot=False)
+    if max_model_name in model_best_predictions:
+        print("\n\n")
+        print("Plot was saved in plots folder")
+        answer = input("Do you want to see them? [y]:") 
+        if answer == "y" or answer == "yes":
+            plot_results(best_predictions, y_test, title, file_name, show_plot=True)
+        else:
+            plot_results(best_predictions, y_test, title, file_name, show_plot=False)
 
     # Save results of predicitons in a file named results.csv
     results_df = pd.DataFrame(columns=[output, "predictions"])
     results_df[output] = y_test
     results_df.rename(columns={output: "actual"}, inplace=True)
-    pred_list = [round(x,2) for x in best_predictions]
-    results_df["predictions"] = pred_list # pd.Series(pred_list)
-    results_df.to_csv("results.csv", index=False)
-    print("Predictions of best model were saved in results.csv")
-    answer = input("Do you want to see them? [y]:") 
-    if answer == "y" or answer == "yes":
-       print("======= Predictions of best model:", best_model_name)
-       print(results_df)
+    if max_model_name in model_best_predictions:
+        pred_list = [round(x,2) for x in best_predictions]
+        results_df["predictions"] = pred_list # pd.Series(pred_list)
+        results_df.to_csv("results.csv", index=False)
+        print("Predictions of best model were saved in results.csv")
+        answer = input("Do you want to see them? [y]:") 
+        if answer == "y" or answer == "yes":
+           print("======= Predictions of best model:", best_model_name)
+           print(results_df)
 
 best_model = models[best_model_index]
 best_model_name = model_names[best_model_index]
 while True:
     print("\n\n")
     print(f"================= Feature Selection ({best_model_name}:{best_epochs} epochs, {best_hidden_sizes}) ======")
     print("\nPlease select a feature selection or sensitivity analysis method:\n")
     print("1. Backward Feature Elimination")
     print("2. Forward Feature Selection")
     print("3. Weight Analysis")
     print("4. Jackknife Sensitivity Analysis (Node Deletion Sensitivity)")
     print("q. Quit")
 
     answer = input("Enter the number of the method you want to run (or 'q' to quit): ").strip().lower()
 
     if answer == '1':
         print("============================= Backward Feature Elimination =============")
         backward_table = backward_feature_elimination(best_model, data, inputs, output, best_epochs, best_hidden_sizes)
         print("------------ backward feature elimination ---------------")
         print(backward_table)
 
         backward_table_latex = generate_latex_table(backward_table, caption="Results of Backward Feature Elimination", label="backward")
         with open(os.path.join("tables", "backward.tex"), 'w', encoding='utf-8') as f:
             print(backward_table_latex, file=f)
 
@@ -587,26 +603,25 @@ while True:
         # Investigate features with negative sensitivity
         negative_sensitivity_features = jackknife_table[jackknife_table['Sensitivity'] < 0]
         print("\nFeatures with negative sensitivity (potentially redundant or harmful):")
         print(negative_sensitivity_features)
 
     elif answer == 'q':
         print("Exiting the feature selection and sensitivity analysis loop. Goodbye!")
         break
     else:
         print("Invalid choice, please try again.")
     print("\n----------------------------------------------------------")
     input("Press any key to return to main menu ...")
 
 
 
 print("----------------------Important! READ -------------------")
 print("latex code for tables are saved in tables folder")
 print("predictions are saved in results.csv file")
 input("Plots have been saved in the 'plots' folder. (press any key to exit)")
 
 # Visualize the model and save it on mlp_structure image
 # dummy_input = torch.randn(1, input_size)
 # from torchviz import make_dot
 # dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
 # dot.render("mlp_structure", format="png", cleanup=True)
-
 