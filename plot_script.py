import json
import matplotlib.pyplot as plt


regular_training_path = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_165858.json"
remove_30_path = r'C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_165059.json'
remove_20_add_50_path = r'C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_170044.json'
remove_20_add_50_path_2 = r'C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_170253.json'
p = r'C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_170501.json'
p2 = r'C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_170507.json'
p3 = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_172235.json"
pp3 = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_172426.json"
ppp3 = r'C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20250925_172653.json'

merged_path = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20251002_145945.json"
reg_p = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20251002_150611.json"
reg_p = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\vfl_test_results_20251002_150801.json"

backtrack = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\house_energy_vfl_test_results_20251014_161019.json"
no_backtrack = r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\run_dumps\house_energy_vfl_test_results_20251014_161043.json"


# Load the JSON results file
with open(no_backtrack, 'r') as f:
    data = json.load(f)

results = data['results']

# Extract training rounds, accuracies, and number of clients
rounds = [entry['train_round'] for entry in results]
accuracies = [entry['accuracy'] for entry in results]
clients = [entry['clients'] for entry in results]

print(len(rounds))

# Plot accuracy over rounds, color by number of clients
plt.figure(figsize=(10, 6))

# Find transition points where number of clients changes
# We'll split into segments with constant number of clients
segments = []
start_idx = 0
for i in range(1, len(clients)):
    if clients[i] != clients[i-1]:
        segments.append((start_idx, i, clients[i-1]))
        start_idx = i
segments.append((start_idx, len(clients), clients[-1]))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for idx, (start, end, n_clients) in enumerate(segments):
    plt.plot(rounds[start:end], accuracies[start:end], label=f"{n_clients} clients", color=colors[idx % len(colors)])

plt.xlabel('Training Round')
plt.ylabel('Accuracy (%)')
plt.title('VFL Training Accuracy Over Rounds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('vfl_accuracy_plot.png')
plt.show()