'''Running this script requires https://github.com/drivelineresearch/openbiomechanics to be installed on
your local machine to access the necessary c3d data'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import c3d
import os
import numpy as np
import math
import re

def calculate_line_direction(df,x1, y1, x2, y2):
    this_angle = []
    for i in range(0,len(df)):
        dx = df[x2][i] - df[x1][i]
        dy = df[y2][i] - df[y1][i]
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        this_angle.append(angle_degrees)
    
    df['dirs'] = this_angle
    return df

def calculate_line_distance(df,x1, y1, x2, y2):
    this_dist = []
    for i in range(0,len(df)):
        dx = df[x2][i] - df[x1][i]
        dy = df[y2][i] - df[y1][i]
        dist = np.sqrt((dx) ** 2 + (dy) ** 2)
        this_dist.append(dist)
    
    df['dist'] = this_dist
    return df

def read_c3d_to_dataframe(file_path):
    with open(file_path, 'rb') as f:
        reader = c3d.Reader(f)
        
        # Prepare lists to store the data
        data = []
        labels = reader.point_labels
        
        # Iterate through each frame
        for i, points, analog in reader.read_frames():
            # Iterate through each marker
            for point_label, point in zip(labels, points):
                x, y, z = point[:3]
                residual = point[3]
                if point_label.strip() in ['Marker1','Marker2','Marker3']:
                    data.append([i, point_label.strip(), x, y, z, residual])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Frame', 'Marker', 'X', 'Y', 'Z', 'Residual'])
    return df

def normalize_together(df):
    flattened = df.values.flatten()
    flattened = flattened.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_flattened = scaler.fit_transform(flattened)
    normalized = normalized_flattened.reshape(df.shape)
    normalized_df = pd.DataFrame(normalized, columns=df.columns)
    return normalized_df


hit_meta = pd.read_csv('penbiomechanics/baseball_hitting/data/metadata.csv')
start_dir = 'openbiomechanics/baseball_hitting/data/c3d/'
all_files = os.listdir(start_dir)

all_3d = []
for i in range(0,len(all_files)):
    print(i)
    if '000' in all_files[i]:
        num = int(all_files[i])
        filt_hit = hit_meta[hit_meta['session']==num].reset_index()
        if len(filt_hit) > 0:
            hit_side = filt_hit.hitter_side[0]
            bl = filt_hit.bat_length_in[0]
            c3d_files = os.listdir(start_dir+all_files[i])
            c3d_path = start_dir+all_files[i]
            if hit_side == 'R':
                for j in range(0,len(c3d_files)):
                    if 'model' not in c3d_files[j]:
                        tab = read_c3d_to_dataframe(c3d_path+'/'+c3d_files[j])
                        if len(tab) > 0:
                            df_wide = tab.pivot_table(index='Frame', columns='Marker', values=['X', 'Y', 'Z'])

                            df_wide.columns = [f'{j}{i}' for i, j in df_wide.columns]

                            df_wide.reset_index(inplace=True)

                            if hit_side == 'L':
                                next
                                print('turning x to negative for leftys')
                                df_wide['Marker2Y'] = -df_wide['Marker2Y']
                                df_wide['Marker3Y'] = -df_wide['Marker3Y']
                                df_wide['Marker1Y'] = -df_wide['Marker1Y']
                                df_wide['Marker2X'] = -df_wide['Marker2X']
                                df_wide['Marker3X'] = -df_wide['Marker3X']
                                df_wide['Marker1X'] = -df_wide['Marker1X']

                            df_wide['Marker2_3X'] = (df_wide['Marker2X'] + df_wide['Marker3X']) / 2
                            df_wide['Marker2_3Y'] = (df_wide['Marker2Y'] + df_wide['Marker3Y']) / 2
                            df_wide['Marker2_3Z'] = (df_wide['Marker2Z'] + df_wide['Marker3Z']) / 2
                            df_wide = calculate_line_direction(df_wide,'Marker1X','Marker1Z','Marker2_3X','Marker2_3Z')
                            df_wide = calculate_line_distance(df_wide,'Marker1X','Marker1Z','Marker2_3X','Marker2_3Z')
                            df_wide['dist'] = df_wide['dist'] / bl
                            all_3d.append(df_wide)


# Normalize the coordinates
df = pd.concat(all_3d)
df = df.reset_index()



scaler = StandardScaler()
coords = df[['Marker1X', 'Marker2_3X','Marker1Y', 'Marker2_3Y', 'Marker1Z', 'Marker2_3Z']]
normalized_df = normalize_together(coords)

# normalized_coords = scaler.fit_transform(coords)
# normalized_df = pd.DataFrame(normalized_coords, columns=coords.columns)
df = pd.concat([df[['Frame']],df[['dirs']], normalized_df], axis=1)

# Prepare the features and target for each marker
def create_features(df, marker):
    features = pd.DataFrame()
    features[f'{marker}X_t-1'] = df[f'{marker}X'].shift(1)
    features[f'{marker}X_t'] = df[f'{marker}X']
    features[f'{marker}X_t+1'] = df[f'{marker}X'].shift(-1)
    features[f'{marker}Z_t-1'] = df[f'{marker}Z'].shift(1)
    features[f'{marker}Z_t'] = df[f'{marker}Z']
    features[f'{marker}Z_t+1'] = df[f'{marker}Z'].shift(-1)
    target = df[f'{marker}Y']
    return features, target

#Create marker features
markers = ['Marker1', 'Marker2_3']
X_list = []
y_list = []
for marker in markers:
    X, y = create_features(df, marker)
    X = X.dropna()
    y = y.loc[X.index]
    X_list.append(X)
    y_list.append(y)

#concat all features
X = pd.concat(X_list,axis=1)
y = pd.concat(y_list,axis=1)

# Ensure there are no NA values
print(X.isna().sum())


#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build NN model
nnmodel = Sequential()
nnmodel.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nnmodel.add(Dense(64, activation='relu'))
nnmodel.add(Dense(y_train.shape[1]))

#Compile
nnmodel.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

#Train
nnmodel.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#Predict
y_pred = nnmodel.predict(X_test)

#Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

nnmodel.save('rh_5ts_model.h5')
