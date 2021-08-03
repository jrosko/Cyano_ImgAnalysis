import matplotlib.pyplot as plt
import numpy as np
import pims
import cv2
from skimage.morphology import skeletonize

def pre_process(input_frame,  thr_val, denoise =  True, op_ker_d = 2):
    """
    Denoise and threshold a frame.
    Uses a non-local means filter to denoise (pass denoise = False to not).
    Applies a binary threshold of thr_val.
    Performs morphological opening to remove any leftover noise patches.
    op_ker_d is the kernel dimension for opening.
    """
    if denoise == True:
        input_frame = cv2.fastNlMeansDenoising(input_frame,templateWindowSize=7, searchWindowSize=21, h=0.7)

    ret_bin, thr_bin = cv2.threshold(input_frame, thr_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((op_ker_d,op_ker_d), np.uint8)
    opening = cv2.morphologyEx(thr_bin, cv2.MORPH_OPEN,kernel, iterations = 1)

    return opening


# test function for pruning
def prune_skeleton(skeleton_input):
    skeleton = np.copy(skeleton_input) # if i dont copy it actually changes first_skeleton
    skeleton = np.pad(skeleton,5, constant_values=(0)) # move this into pre process
    args = np.argwhere(skeleton >0 ).astype('uint8') # indices of non zero pixels
    one_n = [] # simple ends
    three_n = [] # forks
    for arg in args:
        nbors = nearest_neighbors(arg,skeleton)
        if nbors == 3:
            three_n.append(arg)
        elif nbors == 1:
            one_n.append(arg)
    one_n = np.array(one_n).astype(np.int32)
    three_n = np.array(three_n).astype(np.int32)
    ## fork can be two things, a offshoot in the middle of the line or the end fork
    ## let's deal with offshoots first
    porcodio = 0
    for nbor in three_n:
        distances = []
        for p in range(0, len(one_n)):
            dist = np.linalg.norm(nbor - one_n[p], ord=None)
            distances.append([dist,p])
        distances = np.array(distances)
        min_dist_ind = distances[np.argmin(distances[:,0])][1].astype(np.int16)
        min_dist_pt = one_n[min_dist_ind]
        # now we know the point closest to the 3-fork
        # let's begin from this point
        # very nearest nbors should be
        found = 0
        current_pt = min_dist_pt
        while True:
            nearest_pt = nearest(skeleton, current_pt)
            if type(nearest_pt) == type(None):
                porcodio = 1
                break
            if np.all(current_pt == nbor):
                break
            else:
                skeleton[current_pt[0], current_pt[1]] = 0
                current_pt = nearest_pt
        #skeleton[min_dist_pt[0], min_dist_pt[1]]= 0
    if porcodio != 1:
        return skeleton
    else:
        return None

## helper functions

def nearest(image, point):
    very_near = point - np.array([[1,0],[-1,0],[0,1],[0,-1]])
    near = point - np.array([[1,1],[-1,-1],[-1,1],[1,-1]])
    rtn = 0
    found = 0
    if np.any(image[very_near[:,0], very_near[:,1]] > 0):
        for point in very_near:
            if image[point[0], point[1]] > 0:
                rtn = point
                found = 1
                break
    elif np.any(image[near[:,0], near[:,1]] > 0) and found == 0:
        for point in near:
            if image[point[0], point[1]] > 0:
                rtn = point
                found  = 1
                break
    if found == 1:
        return rtn
    else:
        return None

def nearest_neighbors(index_pair,image):
    """
    Takes in the an index pair, belonging to arghwhere(image > 0), and the respective image.
    Interrogates a 3x3 lattice centred around the index_pair, in an anticlockwise way.
    Counts unique neighbours to the central element.
    Returns count of unique members
    """
    delta_i = np.array([[0,1], [0,0], [1,0], [2,0], [2,1], [2,2], [1,2], [0,2]]) - np.array([1,1])
    initial_nonzero = 0
    unique_count = 0
    prev_nonzero = 0
    for m in range(0, len(delta_i)):
        new_i = index_pair + delta_i[m] #
        if image[new_i[0], new_i[1]] > 0: # Can't index image with [new_i] but need to separate the values
            if prev_nonzero == 0:
                unique_count = unique_count + 1
                prev_nonzero = 1
                if m == 0:
                    initial_nonzero = 1
                if m == len(delta_i)-1 and initial_nonzero == 1:
                    unique_count = unique_count - 1
        elif image[new_i[0], new_i[1]] == 0:
            prev_nonzero=0
    return unique_count

def get_skeletons(input_frames,min_mass=50):
    """ returns args of skeleton points in the main image reference frame.
    the output is of kind out['frame']['mass']= xy points"""
    results_dict = {}

    for f in range(0, len(input_frames)):

        image = pre_process(input_frames[f], 9)
        results_sub_dict = {}

        #### NEED TO PREPROCESS OR STH

        # We use connected components with stats to label the image
        retval,labels,stats,centroid=cv2.connectedComponentsWithStats(image)

        # Label candidates
        candidates  = np.argwhere(stats[:,-1]>min_mass).flatten()[1:] # stats[:,-1] is last column, area. candidates[0] is background

        # I want a mask that gives True when a label is not an element of candidates
        filter_mask = np.logical_not(np.isin(labels, candidates))

        # Set non candidates to background value
        labels[filter_mask]=0

        # Loop over good candidates

        for n in range(0, len(candidates)):
            col_ind = stats[candidates][n][0]
            row_ind = stats[candidates][n][1]
            col_w = stats[candidates][n][2]
            row_w = stats[candidates][n][3]
            mass = stats[candidates][n][-1]

            pixel_value = candidates[n] # n-th label will have integer n for it's pixel value
            part_img = labels[row_ind : row_ind + row_w,col_ind : col_ind+col_w] # Bounding box

            #only show pixels that are of "pixel_value"
            pix_val_mask = part_img == pixel_value
            pix_val_mask_inv = np.logical_not(pix_val_mask)

            # set other labels present to background value
            part_img[pix_val_mask_inv] = 0

            part_img = np.uint8(part_img)
            part_img = cv2.dilate(part_img, np.ones((2,2)), iterations = 1)

            skeleton = skeletonize(part_img, method='lee')

            skeleton = prune_skeleton(skeleton)

            if type(skeleton)!=type(None):
                pts = np.argwhere(skeleton>0)
                offset = np.array([row_ind, col_ind])-5
                pts = pts + offset
                results_sub_dict[str(mass)] = pts

        results_dict[str(f)] = results_sub_dict

    return results_dict


def line_len(line, start, end):
    if len(line) <=1:
        return 0
    else:
        sorted_line = []
        for pt in line:
            norm = np.linalg.norm(start - pt)
            sorted_line.append(norm)
        sorted_line = np.array(sorted_line)
        sorted_line = sorted_line[np.argsort(sorted_line)]
        downsampled = np.unique(np.linspace(0 , len(line), int(len(line)/3)).astype(int))
        linesum = 0
        for m in range(1, len(downsampled)):
            linesum += np.linalg.norm(sorted_line[m-1] - sorted_line[m])
        return linesum

# dont use this later - can probably be merged with above
# actually i might not need it at all
def length(sorted_skeleton):
    sorted_skeleton = sorted_skeleton[:,0:2].astype(np.float)
    length = 0
    indices = np.arange(0, len(sorted_skeleton),5).tolist()
    if len(sorted_skeleton)==indices[-1]+1:
        pass
    else:
        indices.append(len(sorted_skeleton)-1)

    for m in range(1, len(indices)):
        norm = np.linalg.norm(sorted_skeleton[indices[m]] - sorted_skeleton[indices[m-1]])

        length += norm
    return length
#####

def order_points(skeleton, frame):
    """ Need to clean this up, loadsa spaghetti code diomio """
    args = np.copy(skeleton)
    image = np.zeros(frame.shape) # fake imahe
    # now i reconstruct skeleton image with args
    for arg in args:
        image[arg[0], arg[1]]=1

    counter = 0
    reference_pt = 0

    for arg in args:
        if fil.nearest_neighbors(arg, image) == 3:
            counter = counter + 1
    for arg in args:
         if fil.nearest_neighbors(arg,image)==1:
             reference_pt = arg
             break

    if counter==0 and type(reference_pt)==type(np.array([])):
        #proceed with analysis
        #find simple ends
        norms = []
        # sort by euclidean norm
        for arg in args:
            norm = np.linalg.norm(arg.astype(np.float)-reference_pt.astype(np.float))
            norms.append(norm)
        norms = np.array(norms)
        normed_pts = np.insert(args, 2, norms, axis=1)
        sorted_column = normed_pts[:,2].argsort()
        # sort the normed pts by the sorted norm colum
        normed_pts = normed_pts[sorted_column]
        return normed_pts
    elif counter>0 or reference_pt==0:
        return None

# Ok so I want a function that takes in a dictionary
def midpoint_trace(input_dict):
    """ Compute a trace of single filament mid points through the image stack """
    outputs=[]
    frames_n = list(input_dict.keys())
    for n in range(1, len(frames_n)):
        filament = input_dict[frames_n[n-1]] # We start from zeroth filament
        ordered_skeleton = order_points(filament, frames[0])
        #
        if type(ordered_skeleton) != type(None):
            mid_index = int(len(ordered_skeleton)/2)
            mid_point  = ordered_skeleton[mid_index][0:2]
            outputs.append([mid_point, int(frames_n[n-1])])
    return np.array(outputs, dtype=object)

# lets make function to compute a speed trace
def speedtrace(mpt,input_dict):
    # need also input dict to have the whole filament for reference
    # Iterate over midpoints
    first_vector = mpt[1][0] - mpt[0][0]
    result = []
    for frame in mpt[:,1].astype(np.int16)[1:]:
        #Get the respective filament from input dict
        filament =  input_dict[str(frame)]
        filament = order_points(filament, frames[0])[:,0:2]

        current_pt = mpt[np.argwhere(mpt[:,1]==frame)[0][0]][0]
        prev_pt =    mpt[np.argwhere(mpt[:,1]==frame)[0][0]-1][0]
        motion_vector = current_pt - prev_pt
        ##calc dt
        dt = (frame  - mpt[:,1][np.argwhere(mpt[:,1]==frame)[0][0]-1])*5 # 5min

        single_result = []
        for point in filament:
            if np.dot(point - prev_pt, motion_vector) > 0:
                if np.sign(np.dot(current_pt - point, motion_vector)) >0:  # ii thought <0 but this works
                    single_result.append(point.tolist())

        #results.append(single_result) #if len(res) == 0: res.append(current_pt.tolist())
        #results now contains a set of ordered trajectories
        segment_len = fil.line_len(np.array(single_result), prev_pt, current_pt)
        result.append(segment_len*np.sign(np.dot(motion_vector, first_vector)))
    return np.array(result)*1.4748/dt

def trajectory_endpoints(mpt, pt_a, pt_b):
    snake = np.array ( [[mpt[n][0][0],mpt[n][0][1]] for n in range(0, len(mpt))] )

    a_dists = []
    b_dists = []
    for point in snake:
        dist_a = np.linalg.norm(point - pt_a)
        dist_b =  np.linalg.norm(point - pt_b)
        a_dists.append([dist_a, point[0], point[1]])
        b_dists.append([dist_b, point[0], point[1]])
    a_dists = np.array(a_dists)
    b_dists = np.array(b_dists)

    ret_a = a_dists[np.argwhere(a_dists[:,0] == np.min(a_dists[:,0]))[0][0]][1:]
    ret_b = b_dists[np.argwhere(b_dists[:,0] == np.min(b_dists[:,0]))[0][0]][1:]


    return ret_a, ret_b

def project(point_a, trajectory):
    a_dists = []
    for point in trajectory:
        dist_a = np.linalg.norm(point[0:2] - point_a)
        a_dists.append([dist_a, point[0], point[1]])
    a_dists = np.array(a_dists)
    ret_a = a_dists[np.argwhere(a_dists[:,0] == np.min(a_dists[:,0]))[0][0]][1:]
    return ret_a
#should just use this function twice for the above block as well
# result is image, but ordered path are arguments


#find args of ordered path where the mpt ends are closest

def find_args(ordered_path, a_clip, b_clip):
    found_a =0
    found_b =0
    for n in range(0, len(ordered_path)):
        point_x = ordered_path[n][0]
        point_y = ordered_path[n][1]
        if point_x == a_clip[0] and point_y == a_clip[1]:
            found_a = n
        elif point_x == b_clip[0] and point_y == b_clip[1]:
            found_b = n
    return found_a,found_b
