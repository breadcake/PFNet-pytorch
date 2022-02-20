import numpy as np
import cv2
import matplotlib.pyplot as plt


def save_correspondences_img(image, warped_image, pts1, pts1_, pred_pts1_, result_name, mace=None):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.polylines(image, np.int32([pts1]), True, (255, 0, 0), 1, cv2.LINE_AA)

    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2RGB)
    cv2.polylines(warped_image, np.int32([pts1_]), True, (0,0,255), 1, cv2.LINE_AA)
    cv2.polylines(warped_image, np.int32([pred_pts1_]), True, (255, 255, 0), 1, cv2.LINE_AA)
    if mace is not None:
        cv2.putText(warped_image, 'MACE=%.2f'%mace, (200, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 10))
    axis1.imshow(image)
    axis1.set_title('Image')
    axis2.imshow(warped_image)
    axis2.set_title('Warped image')
    fig.savefig(result_name, bbox_inches='tight')
    plt.close(fig)

    # print('Wrote file %s' % result_name)