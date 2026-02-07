import matplotlib.pyplot as plt
import numpy as np
import sys
# Example data arrays - replace with your actual data
# Based on the chart, there are 26 scenarios (0-25)
# baseline_values = [2.3, 1.0, 2.1, 2.5, 2.2, 2.8, 4.1, 2.0, 2.2, 0.8, 3.8, 2.1, 3.0, 1.5, 0.8, 2.5, 1.8, 2.0, 1.4, 0.8, 2.5, 1.2, 1.0, 0.8, 2.8]
# ours_values = [2.5, 1.8, 1.9, 3.2, 3.0, 3.2, 2.2, 2.0, 4.8, 3.8, 4.5, 4.2, 3.8, 0.5, 2.8, 1.8, 3.2, 2.2, 0.9, 2.2, 4.2, 4.0, 2.8, 1.0, 4.0]
d1 = """0.9523942978
60
40.48617072
6.680373844
8.343298828
7.24511145
0.01830168257
70
7.776
1.675192701
34.76025616
7.740718724
1.754362167
36"""
d2 = """0.6229908477
30.42376157
42
6.166498933
15.12
25.2
8.642414469
65
12.96
5.201282057
21.6
0.02812878171
0.70543872
100"""

c1 = """
33.62199535
100
100
17.1291637
100
100
55.42592514
100
100
35.90519334
100
100
90.78948773
100"""
c2 = """31.60439281
52.21100847
100
17.1291637
100
100
24.00670686
100
100
34.40001361
100
100
100
100
"""
baseline_values = [float(line.strip()) for line in c1.strip().split('\n')]
ours_values = [float(line.strip()) for line in c2.strip().split('\n')]
# print(np.mean(baseline_values))
# print(np.mean(ours_values))
# sys.exit(0)

# Create scenario numbers (0-25)
scenarios = np.arange(len(baseline_values))

# Set up the figure and axis
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

# Define bar width and positions
bar_width = 0.35
x_pos = np.arange(len(scenarios))

# Create the bars
bars1 = ax.bar(x_pos - bar_width/2, baseline_values, bar_width,
               label='baseline', color='#1f77b4', alpha=1.0)
bars2 = ax.bar(x_pos + bar_width/2, ours_values, bar_width,
               label='ours', color='#ff7f0e', alpha=1.0)

# Customize the chart
# ax.set_xlabel('Scenario #', fontsize=12)
# ax.set_ylabel('Score', fontsize=12)
# ax.set_title('Evaluation Scores in CARLA Benchmark Scenarios', fontsize=14, fontweight='bold')
ax.set_xlabel('Scenario #', fontsize=12)
ax.set_ylabel('Completion', fontsize=12)
ax.set_title('Completion Percentages in CARLA Benchmark Scenarios', fontsize=14, fontweight='bold')

# Set x-axis ticks and labels
ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios)

# Add legend
ax.legend()

# Set y-axis limits to match the original chart
ax.set_ylim(0, 105)

# Add grid for better readability
ax.grid(True, alpha=0.3, axis='y')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
# plt.show()

# Optional: Save the figure
plt.savefig('eval_neat_completion_comp.png', dpi=300, bbox_inches='tight')