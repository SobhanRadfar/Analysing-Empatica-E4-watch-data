import matplotlib.pyplot as plt


def plot_bio_measurements(final_df, tags, ID=1):
    bio_measurements = [
        ('Heart Rate', 'HR', 'Time', 'BPM'),
        ('Blood Volume Pulse', 'BVP', 'Time', 'BVP'),
        ('Electrodermal Activity', 'EDA', 'Time', 'µS'),
        ('Temperature', 'TEMP', 'Time', '°C'),
        ('Inter Beat Intervals', 'IBI_d', 'Time', 's')
    ]

    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    fig.suptitle(f'Bio Measurements of Student {ID}', fontsize=20)

    # Flatten the axes for easy indexing
    axes_flat = axes.flatten()

    for i, (title, column, xlabel, ylabel) in enumerate(bio_measurements):
        ax = axes_flat[i]
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.plot(final_df.loc[final_df['ID'] == ID].Timestamp, final_df.loc[final_df['ID'] == ID][column], color='black', linewidth=0.7)
        ax.grid(False)
        for j in range(0, len(tags[ID-1])):

            if j == 0:
              non_stress = ax.axvspan(final_df.loc[final_df['ID'] == ID].Timestamp.values[0], tags[ID-1][j], facecolor='green', alpha=0.3, label="_"*j + 'Non-Stress')
              stress = ax.axvspan(tags[ID-1][j], tags[ID-1][j+1], facecolor='red', alpha=0.3, label="_"*j + 'Stress')
            elif j == 1:
              non_stress = ax.axvspan(tags[ID-1][j], tags[ID-1][j+1], facecolor='green', alpha=0.3, label="_"*j + 'Non-Stress')
            elif j == 2:
              stress = ax.axvspan(tags[ID-1][j], tags[ID-1][j+1], facecolor='red', alpha=0.3, label="_"*j + 'Stress')
            elif j == 3:
              non_stress = ax.axvspan(tags[ID-1][j], tags[ID-1][j+1], facecolor='green', alpha=0.3, label="_"*j + 'Non-Stress')
            elif j == 4:
              stress = ax.axvspan(tags[ID-1][j], tags[ID-1][j+1], facecolor='red', alpha=0.3, label="_"*j + 'Stress')
            elif j == 5:
              non_stress = ax.axvspan(tags[ID-1][j], final_df.loc[final_df['ID'] == ID].Timestamp.values[-1], facecolor='green', alpha=0.3, label="_"*j + 'Non-Stress')
              break

        ax.legend(fontsize=7)

    # Remove the extra subplot (if there is one)
    if len(axes_flat) > len(bio_measurements):
        fig.delaxes(axes_flat[-1])

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    
def plot_Non_Bio_Measurements(final_df, ID=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 7))
    fig.suptitle('Non-Bio Measurements', fontsize=15)

    ax1.set_title('ACC X', fontsize=8)
    ax2.set_title('ACC Y', fontsize=8)
    ax3.set_title('ACC Z', fontsize=8)

    ax1.set_xlabel('Time', fontsize=8)
    ax2.set_xlabel('Time', fontsize=8)
    ax3.set_xlabel('Time', fontsize=8)

    ax1.set_ylabel('Acceleration', fontsize=8)
    ax2.set_ylabel('Acceleration', fontsize=8)
    ax3.set_ylabel('Acceleration', fontsize=8)

    # Filter the final_df DataFrame only once for the specific ID
    student_data = final_df.loc[final_df['ID'] == ID]

    # Use the filtered DataFrame for plotting
    ax1.plot(student_data.Timestamp, student_data.ACC_x, color='green', linewidth=0.7)
    ax2.plot(student_data.Timestamp, student_data.ACC_y, color='green', linewidth=0.7)
    ax3.plot(student_data.Timestamp, student_data.ACC_z, color='green', linewidth=0.7)

    # Remove the fourth plot (ax4) since we don't have any data to plot
    ax4.axis('off')

    plt.subplots_adjust(hspace=0.4)
    plt.show()
