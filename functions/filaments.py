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
        if nearest_neighbors(arg, image) == 3:
            counter = counter + 1
    for arg in args:
         if nearest_neighbors(arg,image)==1:
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
def midpoint_trace(input_dict,frame):
    """ Compute a trace of single filament mid points through the image stack """
    outputs=[]
    frames_n = list(input_dict.keys())
    for n in range(1, len(frames_n)):
        filament = input_dict[frames_n[n-1]] # We start from zeroth filament
        ordered_skeleton = order_points(filament, frame)
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
        segment_len = line_len(np.array(single_result), prev_pt, current_pt)
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


############


def next_point(end_point_arg, image):
    # Turn this point into an 'image'
    only_point  = np.zeros((image.shape)).astype('uint8')
    only_point[end_point_arg[0], end_point_arg[1]] = 1
    # Dilate this 'image'
    dilated = cv2.dilate(only_point, np.ones((3,3))).astype('uint8')
    # Intersection between the dilated square and the actual image
    intersection  = cv2.bitwise_and(dilated,image)
    # I caught two points, XOR will give me the one that isn't only_point
    next_pt = cv2.bitwise_xor(intersection,only_point)
    return np.argwhere(next_pt>0).tolist()[0]


def ends_junctions(image, kernels):
    # Find all junctions
    junctions = np.zeros(image.shape).astype('uint8')
    for kernel in kernels[1]:
        junctions +=  cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    # Find all line ends
    end_points = np.zeros(image.shape).astype('uint8')
    for kernel in kernels[2]:
        end_points +=  cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    return end_points, junctions


def prune_skeleton(image, kernels, n_pts):
    pruned = np.pad(output_image,5, constant_values=(0))
    original = np.copy(pruned)

    # Remove superfluous corners from skeleton
    for kernel in kernels[0]:
        pruned = pruned  - cv2.morphologyEx(pruned, cv2.MORPH_HITMISS, kernel)
    end_points, junctions = ends_junctions(pruned, kernels)

    # Pruning time - branches
    end_point_args = np.argwhere(end_points > 0).tolist()
    junction_args = np.argwhere(junctions>0).tolist()
    # Note to myself , [1,2] in [[1,2],[3,4]] kind of stuff only reliably works with lists, arrays can give weird results

    for point in end_point_args:
        current_pt = point
        point_buffer = []
        for m in range(0, n_pts):
            if current_pt in junction_args:
                point_buffer = []
                break
            else:
                point_buffer.append(current_pt)
                next_pt = next_point(current_pt, pruned)
                pruned[current_pt[0], current_pt[1]] = 0
                current_pt = next_pt
        if len(point_buffer) > 0:
            for point in point_buffer:
                pruned[point[0], point[1]] = 1

    # This process might produce a superfluous edge or two
    for kernel in kernels[0]:
        pruned = pruned  - cv2.morphologyEx(pruned, cv2.MORPH_HITMISS, kernel)

    """ Below should be written better but yolo now"""
    # Verify that there are no more junctions
    junctions = np.zeros(pruned.shape).astype('uint8')
    for kernel in kernels[1]:
        junctions +=  cv2.morphologyEx(pruned, cv2.MORPH_HITMISS, kernel)

    if len(np.argwhere(junctions > 0)) > 0 :
        print('Junctions still present: ', len(np.argwhere(junctions > 0)))
        # There may be some loops left
        filled_holes = cv2.morphologyEx(pruned, cv2.MORPH_CLOSE, np.ones((3,3)))
        pruned = skeletonize(filled_holes, method = 'lee')
        # Verify that there are no more junctions
        junctions_test = np.zeros(pruned.shape).astype('uint8')
        for kernel in kernels[1]:
            junctions_test +=  cv2.morphologyEx(pruned, cv2.MORPH_HITMISS, kernel)
        if len(np.argwhere(junctions_test > 0)) > 0:
            print('Junctions still present :', len(np.argwhere(junctions_test > 0)))
            return None
        else:
            print('Junctions Resolved')
            return pruned
    else:
        return pruned
    # Need to fix extremity forks

def load_kernels(paths):
    """ Load morphology kernels from files.
    Expects a list of paths of form
    [path_edge, path_junction, path_end]"""

    kernels = []
    for n in range(0,3):
        with open(paths[n], 'rb') as f:
            kernels.append(np.load(f))
    return kernels
