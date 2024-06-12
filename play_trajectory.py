import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter
import numpy as np

final_df = pd.read_csv('final_df.csv')


all_fixes = final_df.copy()
colors = plt.cm.viridis(np.linspace(0, 1, len(all_fixes)))
# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(all_fixes['Marker2_3X_t'], all_fixes['Marker2_3Z_t'], color=colors, label='head x, head y')
plt.scatter(all_fixes['Marker1X_t'], all_fixes['Marker1Z_t'], color=colors, marker='x', label='knob x, knob y')


plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D Trajectory by Frame - Smooth')
plt.legend()

plt.savefig('2d_scatter_plot_smooth_pres.png')
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Initialize lists to store past positions for the tracer
x1_tracer, y1_tracer, z1_tracer = [], [], []
x2_tracer, y2_tracer, z2_tracer = [], [], []

#Update fun
def update(frame):
    ax.cla()  

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])
    ax.set_zlim([1, -1])

    x1, y1, z1 = final_df.iloc[frame][['Marker1X_t', 'Marker1Y_t', 'Marker1Z_t']]
    x2, y2, z2 = final_df.iloc[frame][['Marker2_3X_t', 'Marker2_3Y_t', 'Marker2_3Z_t']]

    x1_tracer.append(x1)
    y1_tracer.append(y1)
    z1_tracer.append(z1)
    x2_tracer.append(x2)
    y2_tracer.append(y2)
    z2_tracer.append(z2)

    ax.scatter(x1, y1, z1, color='r', label='Marker 1')
    ax.scatter(x2, y2, z2, color='b', label='Marker 2_3')

    ax.plot([x1, x2], [y1, y2], [z1, z2], color='g')

    ax.plot(x1_tracer, y1_tracer, z1_tracer, color='r', linestyle='--', alpha=0.6)
    ax.plot(x2_tracer, y2_tracer, z2_tracer, color='b', linestyle='--', alpha=0.6)

    ax.legend()

# Create animation
ani = FuncAnimation(fig, update, frames=len(final_df), interval=200, blit=False)

# Show the animation
plt.show()