import numpy as np

def synch_streams(G2, G3, G4, G5, motion):
    G2 = G2[75:-75]
    G3 = G3[75:-75]
    G4 = G4[75:-75]
    G5 = G5[75:-75]
    
    new_motion = motion[0:-1:6]
    new_motion = motion[75:(75+2880)]
    
    return G2, G3, G4, G5, new_motion

def generate_segments(G2, G3, G4, G5, motion, segment_len, segments_per_trace, output_downsample_factor=2):
    full_run_length = 2880
    indices = np.random.randint(0,(full_run_length - segment_len), size=(segments_per_trace))
    
    images_out = []
    motion_out = []
    
    for i in indices:
        G2_seg = G2[i:(i+segment_len)]
        G3_seg = G3[i:(i+segment_len)]
        G4_seg = G4[i:(i+segment_len)]
        G5_seg = G5[i:(i+segment_len)]
        motion_seg = motion[i:(i+segment_len):output_downsample_factor]
        
        new_image = np.array([[G2_seg],[G3_seg],[G4_seg],[G5_seg]])
        
        images_out.append(new_image)
        
        motion_out.append(np.asarray(motion_seg))
        
    return images_out, motion_out



class Dataset(object):

    def __init__(self, list_of_trials, test_ratio, GCaMP_channels, kinematic_channels, segment_len, output_downsample_factor,  segments_per_trace):
        self.list_of_trials = list_of_trials
        self.test_ratio = test_ratio
        self.train_ratio = 1-self.test_ratio
        self.GCaMP_channels = GCaMP_channels
        self.num_GCaMP_channels = len(GCaMP_channels)
        self.kinematic_channels = kinematic_channels
        self.num_kinematic_channels = len(kinematic_channels)
        self.segment_len = segment_len
        self.output_downsample_factor = output_downsample_factor
        self.segments_per_trace = segments_per_trace

    def create_dataset(self, whiten_input=True):
        self.X = []
        self.y = []

        self.num_trials = 0

        for idx, trial in enumerate(self.list_of_trials):

            G2_temp, G3_temp, G4_temp, G5_temp, motion_temp = synch_streams(trial.G2_AVG, trial.G3_AVG, trial.G4_AVG, trial.G5_AVG, trial.Motion_Conv)
            images, motion = generate_segments(G2_temp, G3_temp, G4_temp, G5_temp, motion_temp, self.segment_len, self.segments_per_trace, self.output_downsample_factor)
            
            for image in images:
                new_image=np.array(image)
                new_image.reshape(1,self.segment_len,self.num_GCaMP_channels)
                self.X.append(new_image)
            
            for entry in motion:
                self.y.append(np.array(entry))

            self.num_trials += 1

        total_rows = self.segments_per_trace*self.num_trials

        self.X = np.array(self.X).reshape(total_rows, 1, self.segment_len, self.num_GCaMP_channels)
        self.y = np.array(self.y)


        self.y = np.expand_dims(self.y, axis=1)

        self.y.reshape(total_rows, 1, int(self.segment_len/self.output_downsample_factor))
        self.y = np.expand_dims(self.y, axis=3)

        if whiten_input:
            self.X = self.X/(np.max(abs(self.X)))

        split = int(total_rows*self.train_ratio)
        self.X_train = self.X[0:split]
        self.X_test = self.X[split:-1]
        self.y_train = self.y[0:split]
        self.y_test = self.y[split:-1]

        return None


def generate_dataset(list_of_trials):
    data = Dataset(list_of_trials, 0.3, ["G1", 'G2', 'G3', 'G4'], ['net_motion'], 112, 2, 400)
    data.create_dataset()

    return [data.X_train, data.y_train, data.X_test, data.y_test]