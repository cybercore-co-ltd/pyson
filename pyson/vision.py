emport cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def get_min_rect(c, resize_ratio):
    """input a contour and return the min box of it"""
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = (box / resize_ratio).astype(np.int32)
    return box


def get_skeleton(img, line_size):
    """ Get skeleton mask of a binary image 
        Arguments:
            img: input image 2d
        Returns:
            binnary mask skeleton
    """

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    size = np.size(img)
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    kernel = np.ones(shape=[line_size, line_size])
    _ = cv2.dilate(skeleton, kernel, iterations=1)
    return _


def convert_mask_to_cell(line_mask):
    """ Convert a mask of lines to cells.
        Arguments:
            line_mask: mask of lines
        Returns:
            a list of cells        
    """

    def is_rect(contour):
        """ Check if a contour is rectangle.
            Arguments:
                contour: contours.
            Returns:
                Boolean value if contour is a rectangle.
        """
        _, _, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 0 and w * h > 0 and area / w * h > 0.6:
            return True
        else:
            return False

    contours, hierarchy = find_contours(line_mask)
    out_cnts = {}
    for ci, (cnt, h) in enumerate(zip(contours, hierarchy)):
        if is_rect(cnt):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            cnt = np.int0(box)
            if h[-1] == -1:
                out_cnts['table_{}'.format(ci)] = cnt
            else:
                out_cnts['cell_{}_table_{}'.format(ci, h[-1])] = cnt
    return out_cnts


def imread(path, to_gray=False, scale=(0, 255)):
    """ Read image given a path
        Arguments:
            path: path to image
            to_gray: convert image to gray
            scale: if scale (0, 255)
        Return: 
            output image
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if scale[1] != 255:
        min_val, max_val = scale
        # after: 0-1
        img = img / 255
        # scale 0 -> (max_val-min_val)
        img = img * (max_val - min_val)
        # scale: min_val -> max_val
        img = img + min_val

    return img

def imwrite(path, img, is_rgb=True):
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, img)

def resize_by_factor(image, factor):
    """ Resize image by a factor
        Arguments:
            image: input image
            factor: the factor by which the image being resized
        Returns:
            output image
    """
    if type(factor) == tuple:
        fx, fy = factor
    elif type(factor) == float:
        fx = fy = factor
    else:
        raise Exception("type of f must be tuple or float")

    return cv2.resize(image, (0, 0), fx=fx, fy=fy)


def resize_by_size(image, size):
    """ Resize image by given size
        Arguments:
            image: input image
            size: the size at which the image being reized

        Returns:
            output image
    """
    return cv2.resize(image, size)


def resize_to_receptive_field(image, receptive_field=256):
    """Resize to the factor of the wanted receptive field.
    Example: Image of size 1100-800 -> 1024-768
    Arguments:
        image: input iamge
        receptive_field:
    Returns:
        resieed image
    """
    new_h, new_w = np.ceil(np.array(image.shape[:2]) / receptive_field).astype(np.int32) * receptive_field
    image = cv2.resize(image, (new_w, new_h))
    return image


def batch_ratio(preds, targets):
    from fuzzywuzzy import fuzz
    rt = []
    for p, t in zip(preds, targets):
        r = fuzz.ratio(p, t)
        rt.append(r)
    return np.mean(rt)


def plot_images(images, cls_true=None, cls_pred=None,
                space=(0.3, 0.3), mxn=None, 
                size=(5,5), dpi=300, max_w = 1500, output_path=None):
    
    if mxn is None:
        n = max(max_w // max([img.shape[1] for img in images]), 1)
        n = min(n, len(images))
        m = len(images)//n 
        #m = m
        m = max(1, m)
        mxn = (m,n)
        print(mxn)
        
    fig, axes = plt.subplots(*mxn)
    fig.subplots_adjust(hspace=space[0], wspace=space[1])
    fig.figsize = size
    fig.dpi=dpi
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='binary')
            if cls_pred is None and cls_true is not None:
                xlabel = "True: {0}".format(cls_true[i])
            elif cls_pred is None and cls_true is not None:
                xlabel = "True: {0}, Pred: {1}".format(
                    cls_true[i], cls_pred[i])
            else:
                xlabel = None
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close()

    
def run_data_init(initers, shuffle):
    for initer in initers:
        data = initer['index']
        if shuffle:
            data = shuffle_by_batch(data, batch_size)
        sess.run(initer['initer'], {initer['x']: data})



def show(inp, size=10, dpi=300, cmap='gray'):
    """
        Input: either a path or image
    """
    if len(inp.shape) == 4:
        inp = inp[0]
    inp = np.squeeze(inp)
    if type(inp) is str:
        assert os.path.exists(inp)
        inp = read_img(inp)
    if size is None:
        size = max(5, inp.shape[1] // 65)
    plt.figure(figsize=(size, size), dpi=dpi)
    plt.imshow(inp, cmap=cmap)
    plt.show()


def find_contours(thresh):
    """
        Get contour of a binary image
            Arguments:
                thresh: binary image
            Returns:
                Contours: a list of contour
                Hierarchy:

    """
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy[0]


def sort_contours(contours, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                             key=lambda b: b[1][i], reverse=reverse))

    return contours, bounding_boxes


def put_text(image, pos, text):
    return cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2)



def visualize_torch_tensor(input_tensor, normalize=True, output_path=None, input_type='onehot'):
    import torch
    
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.detach().cpu().numpy()
        if len(np.unique(input_tensor)) == 2:
            normalize = False
    # if none save in cache
    if output_path is None:
        os.makedirs('cache', exist_ok=True)
        output_path = 'cache/temp.jpg'
    if normalize:
        input_tensor = (input_tensor-input_tensor.min())/(input_tensor.max()-input_tensor.min())
        input_tensor *= 255

    if input_type == 'onehot':
        c,h,w = input_tensor.shape
        color = np.random.choice(256, (c, 3))
        color_tensor = input_tensor[:,None,:,:]*color[:,:,None,None]
        out_tensor = np.zeros([h, w, 3], color_tensor.dtype)
        for i in range(c):
            out_tensor += np.transpose(color_tensor[i], [1,2,0])
        out_tensor = out_tensor *255 /out_tensor.max()
    elif input_type=='image':
        out_tensor = np.transpose(input_tensor, [1,2,0])
    else:
        raise NotImplemented
    cv2.imwrite(output_path, out_tensor)
    print('out-> :', output_path)
