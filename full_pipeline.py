'''Full pipeline from bat detection to animation rendering. Add a video file and try it out!'''

import cv2
import torch
from PIL import Image
import numpy as np
import math
import math
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
from pykalman import KalmanFilter

class vid_to_animation():
    def __init__(self,vid_path,vid_out='test.avi',bat_model=torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt'),three_mod=load_model('rh_model_tune.h5')):
        print(self)

        draw_dfs,prev_frame = self.detect_vid(vid_path,vid_out,bat_model=bat_model)
        starting_hand = draw_dfs[draw_dfs['hx1'] != 0].reset_index()
        all_fixes= self.fix_coords(draw_dfs,starting_hand)
        self.draw_lines(all_fixes,prev_frame,False)
        cv2.imwrite('new_heads_check.jpg',prev_frame)
        height = draw_dfs['img_ht'][0]
        all_fixes.to_csv('cleaned_data.csv',index=False)
        self.create_animation(all_fixes,25,height)


    def draw_longest_line(self,edges, frame_edges, x, y,color,h,w,height,width,past_det):
        # Apply Hough Line Transform
        if len(past_det) > 1:
            past_det = pd.concat(past_det).reset_index()
            past_std = past_det.tail(5)['dirs'].std()
            past_dirs = past_det.tail(3)['dirs'].mean()
            #print('past std',past_std)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            longest_line = None
            max_len = 0

            store_lines = []
            dirs = []
            x1l = []
            y1l = []
            x2l = []
            y2l = []
            for line in lines:
                for x1, y1, x2, y2 in line:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    store_lines.append(length)
                    dirs.append(self.calculate_line_direction(x1,y1,x2,y2))
                    x1l.append(x+x1)
                    x2l.append(x+x2)
                    y1l.append(y+y1)
                    y2l.append(y+y2)
            
            df = pd.DataFrame({'len':store_lines,'dirs':dirs,'x1':x1l,'x2':x2l,'y1':y1l,'y2':y2l})
            last = df.sort_values(by='len').tail(1).reset_index()
            this_dir = last['dirs'][0]

            skip_filter = True
            #df = df.query("dirs > @this_dir-5 and dirs < @this_dir + 5").sort_values(by='len').tail(2).reset_index()
            if len(past_det) > 1 & skip_filter == False:
                for r in range(0,10):
                    if r == 0:
                        df = df.query("dirs > @this_dir-2 and dirs < @this_dir + 2").sort_values(by='len').reset_index()
                        df['total'] = 'total'
                        df = df.groupby('total').mean().reset_index()
                        if abs(past_dirs - df.dirs[0]) < past_std * 5:
                            break
                    else:
                        df = df.query("dirs > @past_dirs-@past_std and dirs < @past_dirs + @past_std").sort_values(by='len')
                        df['total'] = 'total'
                        df = df.groupby('total').mean().reset_index()
                        if abs(past_dirs - df.dirs[0]) > past_std * 2:
                            break
                    past_std += .2
            else:
                df = df.query("dirs > @this_dir-2 and dirs < @this_dir + 2").sort_values(by='len').reset_index()
                df['total'] = 'total'
                df = df.groupby('total').mean().reset_index()
                if len(past_det) > 1:
                    if abs(past_dirs - df.dirs[0]) > past_std * 2:
                        color = color

            #print(color)
            df['color'] = color
            #print('df',df)
            intersections = self.line_intersections_with_bbox(x, y, x+w, y+h, df['x1'][0], df['y1'][0], df['x2'][0], df['y2'][0])
            #print('int',intersections)            
            xmin = intersections[0][0]
            xmax = intersections[1][0]
            ymin = intersections[0][1]
            ymax = intersections[1][1]
            df['x1'] = min(xmin,xmax)
            df['x2'] = max(xmin,xmax)

            if df['y1'][0] > df['y2'][0]:
                df['y1'] = max(ymin,ymax)
                df['y2'] = min(ymin,ymax)
                df['pos'] = 'up'
                #print('barrel up')
            else:
                df['y1'] = min(ymin,ymax)
                df['y2'] = max(ymin,ymax)
                df['pos'] = 'down'
                #print('barrel down')

            df['img_ht'] = height
            df['img_wid'] = width
            fx,fy = self.find_furthest_endpoint(df['x1'][0], df['y1'][0], df['x2'][0], df['y2'][0], width, height)
            df['fx'] = fx
            df['fy'] = fy

            if fx == df['x1'][0]:
                df['cx'] = df['x2'][0]
            else:
                df['cx'] = df['x1'][0]

            if fy == df['y1'][0]:
                df['cy'] = df['y2'][0]
            else:
                df['cy'] = df['y1'][0]

            edges_list = []
            x1_list = []
            y1_list = []
            x2_list = []
            y2_list = []
            for i in range(0,len(df)):
                x1 = df['x1'][i]
                x2 = df['x2'][i]
                y1 = df['y1'][i]
                y2 = df['y2'][i]
                #cv2.line(frame_edges, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 0), 2)
                edges_list.append(frame_edges)
                y1_list.append(y)
                y2_list.append(y2)
                x1_list.append(x)
                x2_list.append(x2)

            return df

    def draw_old_lines(self,edges,x1,x2,y1,y2):
        for i in range(0,len(edges)):
            cv2.line(edges[i], (x1[i] + x2[i], y1[i] + y2[i]), (x1[i] + x2[i], y1[i] + y2[i]), (0, 255, 0), 2)

    def draw_lines(self,df,img,reset=True):
        if reset:
            df = df.reset_index()
        for i in range(0,len(df)):
            x1 = df['x1'][i]
            x2 = df['x2'][i]
            y1 = df['y1'][i]
            y2 = df['y2'][i]
            fx = df['fx'][i]
            fy = df['fy'][i]
            cx = df['cx'][i]
            cy = df['cy'][i]
            color = df['color'][i]
            if color == 'green':
                color = (0, 255, 0)
            elif color == 'purple':
                color = (128, 0, 128)
            else:
                color = (0,0,255)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            cv2.circle(img, (int(fx),int(fy)), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(img, (int(cx),int(cy)), radius=5, color=(0, 0,255), thickness=-1)

    def line_intersections_with_bbox(self,x_min, y_min, x_max, y_max, x1, y1, x2, y2):
        #calc slope and intercept
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
        else:
            #this line is vertical
            m = np.inf
            b = np.inf

        intersections = []

        #check intersection with left edge
        if m != np.inf:
            y_left = m * x_min + b
            if y_min <= y_left <= y_max:
                intersections.append((x_min, y_left))

        #check interesection with the right edge
        if m != np.inf:
            y_right = m * x_max + b
            if y_min <= y_right <= y_max:
                intersections.append((x_max, y_right))

        #check the intersection with the top edge
        if m != 0:
            x_top = (y_min - b) / m
            if x_min <= x_top <= x_max:
                intersections.append((x_top, y_min))

        #check with bottom edge
        if m != 0:
            x_bottom = (y_max - b) / m
            if x_min <= x_bottom <= x_max:
                intersections.append((x_bottom, y_max))

        return intersections

    def calculate_distance(self,x, y, cx, cy):
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    def find_furthest_endpoint(self,x1, y1, x2, y2, width, height):
        #calc center
        cx, cy = width / 2, height / 2
        
        #calc distances
        distance1 = self.calculate_distance(x1, y1, cx, cy)
        distance2 = self.calculate_distance(x2, y2, cx, cy)
        
        #determine further endpoint
        if distance1 > distance2:
            return (x1, y1)
        else:
            return (x2, y2)

    def fix_coords(self,dfs,starting_head,override=True):
        fixed_df = []
        prev_x = 0
        prev_y = 0
        for i in range(0,len(dfs)):
            if i == 0:
                fx = starting_head['fx'][0]
                fy = starting_head['fy'][0]
            else:
                fx = prev_x
                fy = prev_y
            
            this_f = dfs['frame'][i]
            #print(this_f)
            if override:
                closest_det = abs(starting_head['frame'] - this_f)
                min_frame = closest_det.idxmin()
                fx = starting_head['hx1'][min_frame]
                fy = starting_head['hy1'][min_frame]

            x1 = dfs['fx'][i]
            y1 = dfs['fy'][i]
            x2 = dfs['cx'][i]
            y2 = dfs['cy'][i]
            #print(x1,y1)
            distance1 = self.calculate_distance(x1, y1, fx, fy)
            distance2 = self.calculate_distance(x2, y2, fx, fy)
            #print('DISTANCE 1:',distance1)
            #print('DISTANCE 2:',distance2)
            
            if distance1 > distance2:
                print('GREAT! Correct Detction')
                prev_x = x1
                prev_y = y1
                fixed_df.append(dfs.iloc[[i]])
            else:
                print('False Detection....Correcting Coords')
                prev_x = x2
                prev_y = y2
                newfx = dfs['cx'][i]
                newfy = dfs['cy'][i]
                newcx = dfs['fx'][i]
                newcy = dfs['fy'][i]
                dfs['cx'][i] = newcx
                dfs['cy'][i] = newcy
                dfs['fx'][i] = newfx
                dfs['fy'][i] = newfy

                fixed_df.append(dfs.iloc[[i]])
        
        f = pd.concat(fixed_df)
        #f = f.drop(columns=['level_0'])
        #f = f.droplevel('level_0')
        #print(f)

        f = self.calculate_line_direction_df(f,'cx','cy','fx','fy')
        f = self.calculate_line_distance_df(f,'cx','cy','fx','fy')
        return f
    

    def create_features(self,df, marker):
        features = pd.DataFrame()
        features[f'{marker}X_t-1'] = df[f'{marker}X'].shift(1)
        features[f'{marker}X_t'] = df[f'{marker}X']
        features[f'{marker}X_t+1'] = df[f'{marker}X'].shift(-1)
        features[f'{marker}Z_t-1'] = df[f'{marker}Z'].shift(1)
        features[f'{marker}Z_t'] = df[f'{marker}Z']
        features[f'{marker}Z_t+1'] = df[f'{marker}Z'].shift(-1)
        return features

    def detect_vid(self,vid_in,vid_out,bat_model):
        cap = cv2.VideoCapture(vid_in)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(vid_out, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        #start with prev coordinates
        prev_coords_1 = None
        prev_coords_2 = None
        line_dfs = []
        hand_dfs = []
        i = 0
        while cap.isOpened():
            ret, frame2 = cap.read()
            if not ret:
                break
            # if i > 100:
            #     break
            print('FRAME NUMBER: ',i)
            frame2 = cv2.GaussianBlur(frame2, (5, 5), 1.5)
            height, width = frame2.shape[:2]
            # Convert the frame to RGB for YOLO model
            img_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            # Get the bounding box of the bat
            curr_coords = self.get_coords(img_rgb,bat_model=bat_model)
            hand_coords = self.get_coords(img_rgb,'2',bat_model=bat_model)
            hx,hy,hw,hh = hand_coords
            print('HAND COORDS:',hand_coords)
            hand_df = pd.DataFrame({'x1':[hx],'x2':[hx+hw],'y1':[hy],'y2':[hy+hh],'frame':[i]})

            hand_dfs.append(hand_df)
            
            if curr_coords == (0,0,0,0):
                curr_coords = None
                color = 'red'
            else:
                color = 'green'

            if curr_coords is not None:
                x, y, w, h = curr_coords
                
                #update previous coordinates
                prev_coords_2 = prev_coords_1
                prev_coords_1 = curr_coords
            else:
                #if prev coords exist, use them to predict the current coordinates
                if prev_coords_1 is not None and prev_coords_2 is not None:
                    print('before dir')
                    dx, dy = self.calculate_direction_vector(prev_coords_2, prev_coords_1)
                    print('after dir')
                    x, y, w, h = prev_coords_1
                    w += abs(dx)
                    h += abs(dy)
                    x = max(x,0)
                    curr_coords = (int(x), int(y), int(w), int(h))

            if curr_coords is not None:
                x, y, w, h = curr_coords
                stored_w = w
                stored_h = h
                
                #crop image
                cropped_image = frame2[y:y+h, x:x+w]

                #disregard bilateral filtering
                bilateral_filtered = cropped_image
                # Apply bilateral filter
                #bilateral_filtered = cv2.bilateralFilter(cropped_image, d=10, sigmaColor=75, sigmaSpace=75)
                
                #canny edge
                edges = cv2.Canny(bilateral_filtered, 100, 200, apertureSize=3)
                
                #conver to bgr
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                #create empty frame
                frame_edges = np.zeros_like(frame2)
                
                #replace images
                frame_edges[y:y+h, x:x+w] = edges_colored

                if i > 0:
                    #Optical Flow technique, although didn't end up using
                    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    gray_roi = gray[y:y+h, x:x+w]

                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    prev_gray = prev_gray[y:y+h, x:x+w]
                    #calc optical flow in ROI
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
                    #normalize magnitur
                    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    magnitude_normalized = np.uint8(magnitude_normalized)
                    
                    #optical flow canny edge
                    opt_edges = cv2.Canny(magnitude_normalized, 50, 150)

                    try:
                        this_flow = self.draw_longest_line(opt_edges, frame_edges, x, y,color,stored_h,stored_w,height,width,line_dfs)
                    except:
                        print('couldnt draw opt line')
                    try:
                        if len(this_flow) > 0:
                            print('Opt flow angle:',this_flow['dirs'][0])
                        else:
                            print('No opt edges detected')
                    except:
                        print('No opt edges detected')
                
                # Draw the longest straight line on the frame
                try:
                    this_df = self.draw_longest_line(edges, frame_edges, x, y,color,h,w,height,width,line_dfs)
                    this_df['frame'] = i
                    print('longest line angle:',this_df['dirs'][0])
                    line_dfs.append(this_df)
                
                    draw_dfs = pd.concat(line_dfs)

                    self.draw_lines(draw_dfs,frame2)
                except Exception as e:
                    print(e)
                    print('no line detected')

            cv2.imshow('Orig frame with Flow Direction', frame2)
            i += 1
            prev_frame = frame2
            if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'ESC' is pressed
                break

        cap.release()
        cv2.destroyAllWindows()

        hand_data = pd.concat(hand_dfs)
        hand_data = hand_data.rename(columns={'x1':'hx1','x2':'hx2','y1':'hy1','y2':'hy2'})

        draw_dfs = draw_dfs.merge(hand_data,on='frame',how='left')

        return draw_dfs,prev_frame
    

    def get_coords(self,frame2,type='0',bat_model=''):
        results = bat_model(frame2) 
        results = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == type]
        if len(results) == 0:
            return int(0), int(0), int(0), int(0)

        if len(results) > 0:
            results = results.reset_index()

            x = results['xmin'][0]
            y = results['ymin'][0]
            w = results['xmax'][0]-results['xmin'][0]
            h = results['ymax'][0]-results['ymin'][0]
        else:
            x = 0
            y = 0
            w = 1
            h = 1
        return int(x), int(y), int(w), int(h)
    

    def calculate_direction_vector(self,coords1, coords2):
        x1, y1, _, _ = coords1
        x2, y2, _, _ = coords2
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy

    def calculate_line_direction(self,x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees
    

    def normalize_angle(self,angle):
        return angle % 360

    def calculate_line_direction_df(self,df, x1, y1, x2, y2):
        angles = []
        for i in range(len(df)):
            dx = df[x2][i] - df[x1][i]
            dy = df[y2][i] - df[y1][i]
            angle_radians = math.atan2(dy, dx)
            angle_degrees = math.degrees(angle_radians)
            normalized_angle = self.normalize_angle(angle_degrees)
            angles.append(normalized_angle)
        
        df['dirs'] = angles
        return df
    
    def calculate_line_distance_df(self,df,x1, y1, x2, y2):
        this_dist = []
        for i in range(0,len(df)):
            dx = df[x2][i] - df[x1][i]
            dy = df[y2][i] - df[y1][i]
            dist = np.sqrt((dx) ** 2 + (dy) ** 2)
            this_dist.append(dist)
        
        df['dist'] = this_dist
        return df

    def interp_fun(self,data, cols, num, time_field):
        # crea
        new_time = np.linspace(data[time_field].min(), data[time_field].max(), num=num)
        
        #interpolated columns
        interpolated_cols = []
        
        #interpolate each column
        for col in cols:
            interpolated_col = np.interp(new_time, data[time_field], data[col])
            interpolated_cols.append(interpolated_col)
        
        # Combine the interpolated columns into a DataFrame
        interpolated_data = pd.DataFrame(interpolated_cols).T
        
        #Rename the columns
        interpolated_data.columns = cols
    
        return interpolated_data
    

    def create_animation(self,all_fixes,window_size,height):
        # Define the window size and polynomial order for the Savitzky-Golay filter
        window_size = window_size  
        poly_order = 2

        '''Plotting 2D Coordinates'''
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_fixes)))
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(all_fixes['fx'], all_fixes['fy'], color=colors, label='fx, fy')
        plt.scatter(all_fixes['cx'], all_fixes['cy'], color=colors, marker='x', label='cx, cy')

        # Add labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('2D Trajectory by Frame - Actual Detections')
        plt.legend()
        #plt.show()
        plt.savefig('2d_scatter_plot_actual.png')
        #plt.close()
        # Apply the Savitzky-Golay filter to smooth the fx, fy, cx, cy columns
        
        #scaler = StandardScaler()
        scaler = MinMaxScaler()
        all_fixes['fx'] = savgol_filter(all_fixes['fx'], window_length=window_size, polyorder=poly_order)
        all_fixes['fy'] = savgol_filter(all_fixes['fy'], window_length=window_size, polyorder=poly_order)
        all_fixes['cx'] = savgol_filter(all_fixes['cx'], window_length=window_size, polyorder=poly_order)
        all_fixes['cy'] = savgol_filter(all_fixes['cy'], window_length=window_size, polyorder=poly_order)

        all_fixes = self.calculate_line_direction_df(all_fixes,'cx','cy','fx','fy')
        all_fixes['rolling_dir'] = all_fixes['dirs'].rolling(window=5).mean()
    
        num_frames = 600

        cols = ['fx','fy','cx','cy','dirs','rolling_dir']
        #Perform interpolation
        all_fixes = self.interp_fun(all_fixes, cols, num_frames, 'frame')
        all_fixes['frame'] = all_fixes.index
        all_fixes.to_csv('before_clean.csv',index=False)
        '''Plotting 2D Coordinates'''
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_fixes)))
        #scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(all_fixes['fx'], all_fixes['fy'], color=colors, label='fx, fy')
        plt.scatter(all_fixes['cx'], all_fixes['cy'], color=colors, marker='x', label='cx, cy')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('2D Trajectory by Frame - Smooth')
        plt.legend()
        plt.show()
        plt.savefig('2d_scatter_plot_smooth.png')
        plt.close()


        def normalize_together(df):
            flattened = df.values.flatten()
            flattened = flattened.reshape(-1, 1)
            scaler = MinMaxScaler()
            normalized_flattened = scaler.fit_transform(flattened)
            normalized = normalized_flattened.reshape(df.shape)
            normalized_df = pd.DataFrame(normalized, columns=df.columns)
            return normalized_df

        cleaned = all_fixes
        cleaned = cleaned.rename(columns={'fx':'Marker2_3X','fy':'Marker2_3Z','cx':'Marker1X','cy':'Marker1Z'})
        cleaned = cleaned[['Marker1X', 'Marker2_3X','Marker1Z', 'Marker2_3Z']]
        cleaned['Marker2_3Z'] = cleaned['Marker2_3Z']
        cleaned['Marker1Z'] = cleaned['Marker1Z']
        normalized_df = normalize_together(cleaned)
        # normalized_coords = scaler.fit_transform(cleaned)
        # normalized_df = pd.DataFrame(normalized_coords, columns=cleaned.columns)
        df = pd.concat([all_fixes[['frame']],all_fixes['dirs'], normalized_df], axis=1)

        '''Plotting 2D Coordinates Normalized'''
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_fixes)))
        plt.figure(figsize=(10, 6))
        plt.scatter(normalized_df['Marker2_3X'], normalized_df['Marker2_3Z'], color=colors, label='fx, fy')
        plt.scatter(normalized_df['Marker1X'], normalized_df['Marker1Z'], color=colors, marker='x', label='cx, cy')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('2D Trajectory by Frame - Smooth Normalized')
        plt.legend()
        plt.show()
        plt.savefig('2d_scatter_plot_normalized.png')
        plt.close()

        markers = ['Marker1', 'Marker2_3']
        X_list = []
        y_list = []
        for marker in markers:
            X = self.create_features(df, marker)
            X = X.dropna()
            X_list.append(X)

        # Concatenate all markers' data
        X = pd.concat(X_list,axis=1)
        print(X.isna().sum())

        nnmodel = load_model('rh_5ts_model.h5')

        y_pred = nnmodel.predict(X)
        y1 = y_pred[:, 0]
        y2_3 = y_pred[:,1]
        
        final_df = X[['Marker1X_t','Marker1Z_t','Marker2_3X_t','Marker2_3Z_t']]

        final_df['Marker1Y_t'] = y1
        final_df['Marker2_3Y_t'] = y2_3

        window_size = 49
        final_df['Marker1Y_t'] = savgol_filter(final_df['Marker1Y_t'], window_length=window_size, polyorder=poly_order)
        final_df['Marker2_3Y_t'] = savgol_filter(final_df['Marker2_3Y_t'], window_length=window_size, polyorder=poly_order)

        print('final df',final_df)
        final_df.to_csv('final_df.csv',index=False)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])

        x1_tracer, y1_tracer, z1_tracer = [], [], []
        x2_tracer, y2_tracer, z2_tracer = [], [], []

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

            ax.scatter(x1, y1, z1, color='r', label='Marker 1' if frame == 0 else "")
            ax.scatter(x2, y2, z2, color='b', label='Marker 2/3' if frame == 0 else "")

            ax.plot([x1, x2], [y1, y2], [z1, z2], color='g')

            ax.plot(x1_tracer, y1_tracer, z1_tracer, color='r', linestyle='--', alpha=0.6)
            ax.plot(x2_tracer, y2_tracer, z2_tracer, color='b', linestyle='--', alpha=0.6)

            if frame == 0:
                ax.legend()

        ani = FuncAnimation(fig, update, frames=len(final_df), interval=200, blit=False)

        plt.show()
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('animation.mp4', writer='ffmpeg', fps=30)


    def implement_kalman(self,df,field,new_field):
        initial_state = df[field].iloc[0]  
        transition_matrix = [[1]] 
        observation_matrix = [[1]] 
        initial_state_covariance = 1 
        transition_covariance = 1 
        observation_covariance = 1

        kf = KalmanFilter(
            initial_state_mean=initial_state,
            initial_state_covariance=initial_state_covariance,
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance
        )

        state_means, state_covariances = kf.smooth(df[field].values)

        # Add the smoothed values to the DataFrame
        df[new_field] = state_means
        return df

if __name__ == "__main__":
    vid_to_animation(vid_path='tee_work_trim.mov',vid_out='test.avi')
