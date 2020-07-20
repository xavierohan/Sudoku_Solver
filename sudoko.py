import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms




def sudoko_ref_grid():

    row_idx = 'ABCDEFGHI'
    column_idx = '123456789'

    rows = []
    columns = []
    box = []

    coords = [a + b for a in row_idx for b in column_idx]

    for i in row_idx:
        rows.append([i + j for j in column_idx])
    for i in column_idx:
        columns.append([j + i for j in row_idx])
    for i in ('ABC', 'DEF', 'GHI'):
        for j in ('123', '456', '789'):
            box.append([h + k for h in i for k in j])

    all_units = rows + columns + box
    groups = {}
    groups['units'] = {pos: [unit for unit in all_units if pos in unit] for pos in coords}

    groups['peers'] = {pos: set(sum(groups['units'][pos], [])) - {pos} for pos in coords}


    return coords, groups, all_units


def parse_puzzle(puzzle, digits='123456789', nulls='0.'):
    flattened_puzzle = ['.' if char in nulls else char for char in puzzle if char in digits + nulls]
    if len(flattened_puzzle) != 81:
        raise ValueError(
            'Input puzzle has %s grid positions specified, must be 81. Specify a position using any ''digit from 1-9 and 0 or . for empty positions.' % len(flattened_puzzle))
    coords, groups, all_units = sudoko_ref_grid()

    return dict(zip(coords, flattened_puzzle))


def validate(puzzle):
    if puzzle is None:
        return False
    coords, groups, all_units = sudoko_ref_grid()
    full = [str(x) for x in range(1, 10)]
    return all([sorted([puzzle[cell] for cell in unit]) == full for unit in all_units])


puzzle1 = """
   	 1  .  5 | .  7  . | 4  .  .
   	 .  8  . | 2  .  . | .  .  .
   	 7  2  4 | .  .  1 | .  .  6
   	---------+---------+---------
   	 .  .  . | 3  2  5 | .  .  .
   	 2  3  7 | .  .  . | 1  4  5
   	 6  .  . | 4  1  7 | .  .  .
   	---------+---------+---------
   	 8  .  . | 1  .  . | 6  2  4
   	 .  .  . | .  .  3 | .  5  .
   	 .  .  1 | .  4  . | 3  .  9
   	"""
puzzle2 = """
 8  .  . | .  .  . | .  .  .
 .  .  3 | 6  .  . | .  .  .
 .  7  . | .  9  . | 2  .  .
---------+---------+---------
 .  5  . | .  .  7 | .  .  .
 .  .  . | .  4  5 | 7  .  .
 .  .  . | 1  .  . | .  3  .
---------+---------+---------
 .  .  1 | .  .  . | .  6  8
 .  .  8 | 5  .  . | .  1  .
 .  9  . | .  .  . | 4  .  .
"""

# k = parse_puzzle(puzzle1)
# print(k)


def solve_puzzle(puzzle):
    digits = '123456789'
    coords, groups, all_unts = sudoko_ref_grid()
    input_grid = parse_puzzle(puzzle)
    input_grid = {k: v for k, v in input_grid.items() if v != '.'}
    output_grid = {cell: digits for cell in coords}

    def confirm_value(grid, pos, val):  # by cofirming a value we can delete all instances of that value in its peer
        rem_val = grid[pos].replace(val, '')
        for val in rem_val:
            grid = eliminate(grid, pos, val)
        return grid

    def eliminate(grid, pos, val):
        if grid is None:
            return None
        if val not in grid[pos]:  # if already eliminated - do nothing
            return grid
        grid[pos] = grid[pos].replace(val, '')

        if len(grid[pos]) == 0:
            return None

        elif len(grid[pos]) == 1:
            for peer in groups['peers'][pos]:
                grid = eliminate(grid, peer, grid[pos])  # RECURSIVE (used for constraint propagation)
                if grid is None:
                    return None

        for unit in groups['units'][pos]:
            possibilities = [p for p in unit if val in grid[p]]

            if len(possibilities) == 0:
                return None
            elif len(possibilities) == 1 and len(grid[possibilities[0]]) > 1:
                if (confirm_value(grid, possibilities[0], val)) is None:
                    return None

        return grid

    for position, value in input_grid.items():  # for each value given, confirm the value
        output_grid = confirm_value(output_grid, position, value)

    if validate(output_grid):
        return output_grid

    def guess_digit(grid):
        if grid is None:
            return None
        if all([len(possibilities) == 1 for cell, possibilities in grid.items()]):
            return grid
        n, pos = min([(len(possibilities), cell) for cell, possibilities in grid.items() if len(possibilities) > 1])

        for val in grid[pos]:
            solution = guess_digit(confirm_value(grid.copy(), pos, val))
            if solution is not None:
                return solution

    output_grid = guess_digit(output_grid)
    return output_grid

    return output_grid


def disp_puz(x):
    f = list(x.values())
    k = 1
    for i in f:
        print(i, end=' ')
        if k % 3 == 0:
            print('|', end='')
        if k in [9, 18, 27, 36, 45, 54, 63, 72, 81]:
            print('\n', end='')
        if k % 27 == 0:
            print('----------------------')
        k += 1


# k = solve_puzzle(puzzle1)
# f = list(k.values())
#
# disp_puz(k)
# sudoko_ref_grid()









img = cv2.imread('sudoko/30.jpg', cv2.IMREAD_GRAYSCALE)
# print(type(img))
# print(img.shape)
# print(img)
#img = cv2.imread('sudoko/og.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('sudoku puzzle ', img)  # Show the image
cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
cv2.destroyAllWindows()

threshold2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3) #11
#threshold2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 87, 9)

# plt.imshow(threshold2,'gray')
# plt.show()







def preprocess_image(img, skip_dilate=False):
    im1 = np.array(img)
    b = cv2.GaussianBlur(im1, (9, 9), 0)
    b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)  # 11 3 # TRY OPTIMIZING THIS
    #b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    b = cv2.bitwise_not(b, b)
    return b


processed = preprocess_image(img)
cv2.imshow('processed puzzle ', processed)
cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
cv2.destroyAllWindows()
plt.show()

ext_contours, hier = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, hier = cv2.findContours(processed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# Draw all of the contours on the image in 2px red lines
# all_contours = cv2.drawContours(processed.copy(), contours, -1, (255, 0, 0), 2)
external_only = cv2.drawContours(processed.copy(), ext_contours, -1, (255, 0, 0), 2)

# cv2.imshow('d', external_only)
# cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
# cv2.destroyAllWindows()

import operator


def find_corners(img):
    contours_ext, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours_ext, key=cv2.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]  # largest

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


corners = find_corners(processed)


def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    side = max([distance_between(bottom_right, top_right), distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left), distance_between(top_left, top_right)])
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, m, (int(side), int(side)))
###########


#############
cropped = crop_and_warp(img, corners)
cv2.imshow('cropped ', cropped)
cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
cv2.destroyAllWindows()


def rect_box(img):
    side = img.shape[:1]
    side = side[0] / 9
    squares = []
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares


rect = rect_box(cropped)


def display_rect_box(x, rects, colour=255):
    f = x.copy()
    for i in rects:
        f = cv2.rectangle(f, tuple(int(x) for x in i[0]), tuple(int(x) for x in i[1]), colour)
    cv2.imshow('bounding box ', f)
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()


display_rect_box(cropped, rect)


def cut_from_rect(img, rect):
    # return img[ int(rect[0][0]):int(rect[1][0]) , int(rect[0][1]):int(rect[1][1])]
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, bg=0):
    h, w = img.shape[:2]

    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)
    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, bg)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_bg=None):
    img = inp_img.copy()
    ht, wd = img.shape[:2]
    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]
    if scan_bg is None:
        scan_bg = [wd, ht]

    for x in range(scan_tl[0], scan_bg[0]):
        for y in range(scan_tl[1], scan_bg[1]):
            # Only operate on light or white squares
            if img.item(y, x) == 255 and x < wd and y < ht:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    for x in range(wd):
        for y in range(ht):
            if img.item(y, x) == 255 and x < wd and y < ht:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((ht + 2, wd + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = ht, 0, wd, 0

    for x in range(wd):
        for y in range(ht):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            # Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]

    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    digit = cut_from_rect(img, rect)
    # print(digit.shape[:2])
    # h, w = 36,35
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    w1 = bbox[1][0] - bbox[0][0]
    h1 = bbox[1][1] - bbox[0][1]

    if w1 > 0 and h1 > 0 and (w1 * h1) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


def get_digits(img, squares, size):
    digits = []
    img = preprocess_image(img.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


squares = rect_box(cropped)
digit = get_digits(cropped, squares, 28)


def show_dig(digits, colour=255):
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    cv2.imshow('dig', np.concatenate(rows))
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()


show_dig(digit)

hidden_neurons = 16
#transform3 = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(),  transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform = transforms.Compose([preprocess_image, transforms.ToPILImage(),transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform2 = transforms.Compose([preprocess_image, transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#transform = transforms.Compose([preprocess_image, transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
#transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform2)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform2)
trainset = trainset + valset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(len(trainset), len(valset))
print(images.shape)
print(labels.shape)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10
from torch import nn, optim

model = torch.load('plswork.pt') #plswork
'''
#model = torch.load('./my_mnist_model2(epoch 15, train+val).pt')
model.eval()
# model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[1], output_size),
#                       nn.LogSoftmax(dim=1)) #1
# print(model)

##############################################################

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
#print(images[0], len(images[0]), images[0].size())
logps = model(images)  # log probabilities
loss = criterion(logps, labels)  # calculate the NLL loss
#
# print('Before backward pass: \n', model[0].weight.grad)
# loss.backward()
# print('After backward pass: \n', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
#
# epochs = 15
# #
# # for e in range(epochs):
# #     running_loss = 0
# #     for images, labels in trainloader:
# #         # Flatten MNIST images into a 784 long vector
# #         images = images.view(images.shape[0], -1)
# #
# #         # Training pass
# #         optimizer.zero_grad()
# #
# #         output = model(images)
# #         loss = criterion(output, labels)
# #
# #         # This is where the model learns by backpropagating
# #         loss.backward()
# #
# #         # And optimizes its weights here
# #         optimizer.step()
# #
# #         running_loss += loss.item()
# #     else:
# #         print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
# # print("\nTraining Time (in minutes) =", (time() - time0) / 60)
# #
# #
# # correct_count, all_count = 0, 0
# # for images, labels in valloader:
# #     for i in range(len(labels)):
# #         img = images[i].view(1, 784)
# #         with torch.no_grad():
# #             logps = model(img)
# #
# #         ps = torch.exp(logps)
# #         probab = list(ps.numpy()[0])
# #         pred_label = probab.index(max(probab))
# #         true_label = labels.numpy()[i]
# #         if (true_label == pred_label):
# #             correct_count += 1
# #         all_count += 1
# #
# # print("Number Of Images Tested =", all_count)
# # print("\nModel Accuracy =", (correct_count / all_count))

#torch.save(model, './my_mnist_model2(epoch 15, train+val).pt')
#torch.save(model,'./my2222_mnist_model2222.pt' )


img = images[0].view(1, 784)

#print('IMAGES[0] , TRANSFORM2(digits[5]))','\n',images[0], transform2(digit[0]).reshape([1, 784]))
# model(transform2(digit[3]).flatten())


def ret_prob(dig):

    with torch.no_grad():
        logps = model(transform2(dig).reshape([1, 784]))
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        return probab
def ret_prob2(dig):
    if dig.sum() == 0:
        return 0
    else:
        with torch.no_grad():
            logps = model(transform2(dig).reshape([1, 784]))
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            return probab
def ret_logprob(dig):
    with torch.no_grad():
        logps = model(transform(dig).reshape([1, 784]))
    return logps


s = 4
# print('7','\n', ret_prob(digit[6]))
# print('1 but 7','\n', ret_prob(digit[8]))
# print('1 but 7 ','\n', ret_prob(digit[18]))
# print('1 but 7 ','\n', ret_prob(digit[59]))
# print('1 but 7 shown','\n', ret_prob(digit[29]))
#
# print('7','\n', ret_logprob(digit[6]))
# print('1 but 7','\n', ret_logprob(digit[8]))
# print('1 but 7 ','\n', ret_logprob(digit[18]))
# print('1 but 7 ','\n', ret_logprob(digit[59]))
# print('1 but 7 shown','\n', ret_logprob(digit[29]))






# p = digit[6].reshape([1,784])[0]
# print(sum(p[0:170]))
#
# q = digit[8].reshape([1,784])[0]
# print(sum(q[0:170]))
#
# q = digit[18].reshape([1,784])[0]
# print(sum(q[0:170]))

################################ save templates of numbers to compare against (ex - np.sum(1) limited to a certain pixel gives lower value than that of 7

#exit()
# p[0:250] =  0
# q[0:250] = 0
# #p[600:784] =  0
# #q[600:784] = 0
# print(ret_prob(p), ret_prob(q))
#
# cv2.imshow('d',digit[6].reshape([28,28]))
# cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
# cv2.destroyAllWindows()

#exit()






def pred_digit(dig):

    if dig.sum() == 0:
        #return 0
        return '.'
    else:
        with torch.no_grad():
            logps = model(transform2(dig).reshape([1, 784]))

        ps = torch.exp(logps)
        #print(ps.numpy())
        probab = list(ps.numpy()[0])
        #probab = list(ps.numpy())
        #print("Predicted Digit =", probab.index(max(probab)))
        a=probab.index(max(probab))
        # if a == 5:
        #     if logps[0][5] < -1.5*10**-6:
        #         a = 6
        if a == 5:
            p = dig.reshape([1, 784])[0]
            p[0:270] = 0
            q = ret_prob(p)
            if q[0] > 0.6:
                a = 6


        if a == 3:
            if probab[3] < 0.98 and probab[8] > 1*10**-15:
                a = 8
            # elif probab[8] > 1*10**-15:
            #     a = 8

        if a == 1 or a==7:
            p = dig.reshape([1,784])[0]
            if(np.sum(p[0:170])) <9000:
                a = 1
            else:
                a = 7

        print(a, probab)




        return str(a)



pred_digit_list=[]
q = []

for i in range(81):
    #q.append(ret_prob2(digit[i]))
    pred_digit_list.append(pred_digit(digit[i]))
print(s)


print(' '.join(pred_digit_list))

puzzle = ' '.join(pred_digit_list)

#print(q)
parsed = parse_puzzle(puzzle)
# print(g.items())





coords, groups, all_units = sudoko_ref_grid()
print(coords, groups, all_units)
print(groups)
peers = groups['peers']
print(peers['A1'])

invalid = []
invalid_pos =[]
def check(p):
    c =0

    k = parse_puzzle(p)
    for i in coords:
        if k[i] != '.':
            for j in peers[i]:
                if k[i] == k[j]:
                    print(i, 'invalid')
                    invalid_pos.append(c)

                    invalid.append([i, j])
        c+=1



check(puzzle)

################# clean the code , make it more efficient
#x = np.array(invalid)
print('invalid', invalid)
seen = set()
unique = []
for x in invalid:
    srtd = tuple(sorted(x))
    if srtd not in seen:
        unique.append(x)
        seen.add(srtd)
print(unique)
#y = np.array(invalid_pos)


for i in unique:
    print(parsed[i[0]], parsed[i[1]])

#print(invalid)

#exit()

# for position, value in k.items():
#     print(position, value)
#
# for position, value in g.items():
#     print(position, value)





k = 1
for i in pred_digit_list:
    print(i, end=' ')

    if k % 3 == 0:
        print('|', end='')
    if k in [9, 18, 27, 36, 45, 54, 63, 72, 81]:
        print('\n', end='')
    if k % 27 == 0:
        print('----------------------')
    k += 1

# cv2.imshow('d', cropped)
# cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
# cv2.destroyAllWindows()
#
# def get_img_corr_label(n):
#     for i in range(1000):
#         if labels[i] == n:
#             #print(n,images[i])
#             return images[i]
#             break
def if_invalid():
    k = parse_puzzle(puzzle)
    print(k)
    for i in unique:
        print(k[i[0]])


def solve_if_valid():

    if len(invalid) == 0:
        temp = solve_puzzle(puzzle)
        disp_puz(temp)
        solved_list = list(temp.values())

        print(rect[0], rect[1])
        list_pos = [index for index, value in enumerate(pred_digit_list) if value == '.']
        print(list_pos)

        for i in list_pos:
            x_mid = (rect[i][0][0] + rect[i][1][0]) / 2
            y_mid = (rect[i][0][1] + rect[i][1][1]) / 2
            cv2.putText(cropped, solved_list[i], (int(x_mid - 5), int(y_mid + 10)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('l', cropped)
        cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
        cv2.destroyAllWindows()

    else:
        if_invalid()


solve_if_valid()

exit()
temp = solve_puzzle(' '.join(s))
disp_puz(temp)
solved_list = list(temp.values())

print(rect[0],rect[1])
list_pos = [index for index, value in enumerate(s) if value == '.']
print(list_pos)

for i in list_pos:
    x_mid = (rect[i][0][0] + rect[i][1][0])/2
    y_mid = (rect[i][0][1] + rect[i][1][1])/2
    cv2.putText(cropped, solved_list[i], (int(x_mid -5), int(y_mid+10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

cv2.imshow('l', cropped)
cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
cv2.destroyAllWindows()

'''
########################


