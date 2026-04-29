import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('output/q3', exist_ok=True)

# Images loaded at 1/4 resolution as specified in the assignment
im1 = cv2.imread('a2_images/c1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
im2 = cv2.imread('a2_images/c2.jpg', cv2.IMREAD_REDUCED_COLOR_4)


# Q3(a) - Compute homography from manually clicked points and warp c1 to c2
#
# Corresponding points were collected using Listing 1 (listing1_click_points.py),
# the interactive click tool provided in the assignment.
# 6 landmarks clicked on both images in the same order:
# TL corner hole, USB connector, BR corner hole, BL corner hole, DC jack, red LED

pts1 = np.array([
    [ 67,  38],
    [460,  85],
    [480, 600],
    [ 55, 590],
    [192,  58],
    [278, 372],
], dtype=np.float32)

pts2 = np.array([
    [ 50,  22],
    [470,  60],
    [500, 550],
    [ 30, 545],
    [170,  42],
    [262, 337],
], dtype=np.float32)

h, w = im2.shape[:2]

H_manual, _ = cv2.findHomography(pts1, pts2, method=0)
warped_manual = cv2.warpPerspective(im1, H_manual, (w, h))

cv2.imwrite('output/q3/q3a_warped_manual.png', warped_manual)
print("Q3(a) Homography (manual):")
print(H_manual)


# Q3(b) - Difference image: reveals misalignments between the two boards

diff_manual = cv2.absdiff(im2, warped_manual)
cv2.imwrite('output/q3/q3b_diff_manual.png', diff_manual)
print("\nQ3(b) saved: output/q3/q3b_diff_manual.png")


# Q3(c) - SIFT keypoints, descriptors, and matches

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# FLANN-based matcher with Lowe's ratio test
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = [m for m, n in matches if m.distance < 0.75 * n.distance]
print(f"\nQ3(c) SIFT: {len(kp1)} keypoints in c1, {len(kp2)} in c2")
print(f"      Good matches after ratio test: {len(good)}")

match_img = cv2.drawMatches(im1, kp1, im2, kp2, good[:50], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('output/q3/q3c_sift_matches.png', match_img)
print("      Saved: output/q3/q3c_sift_matches.png")


# Q3(d) - Homography from SIFT matches, warp, and difference

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H_sift, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(f"\nQ3(d) SIFT homography inliers: {int(mask.sum())}/{len(good)}")
print("Homography (SIFT):")
print(H_sift)

warped_sift = cv2.warpPerspective(im1, H_sift, (w, h))
diff_sift = cv2.absdiff(im2, warped_sift)

cv2.imwrite('output/q3/q3d_warped_sift.png', warped_sift)
cv2.imwrite('output/q3/q3d_diff_sift.png', diff_sift)
print("      Saved: output/q3/q3d_warped_sift.png")
print("      Saved: output/q3/q3d_diff_sift.png")


# Comparison figure: manual vs SIFT results side by side
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Reference: c2')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(warped_manual, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Q3(a): Warped c1 (manual points)')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(diff_manual, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Q3(b): Difference (manual)')
axes[0, 2].axis('off')

axes[1, 0].imshow(cv2.cvtColor(cv2.drawMatches(
    im1, kp1, im2, kp2, good[:30], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS), cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'Q3(c): SIFT matches ({len(good)} total)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(warped_sift, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Q3(d): Warped c1 (SIFT)')
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(diff_sift, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Q3(d): Difference (SIFT)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('output/q3/q3_comparison.png', dpi=130)
plt.close()
print("\nSaved: output/q3/q3_comparison.png")
