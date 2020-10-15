import numpy as np
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    scores_symmetry = np.loadtxt("cfg/tools/data/scores_symmetry")
    scores_features = pickle.load(open("cfg/tools/data/scores_features", "rb"))
    rot_similarity = pickle.load(open("cfg/tools/data/rot_similarity", "rb"))
    # print(len(scores_features))
    # print(len(scores_symmetry))
    scores = {}
    for k, v in rot_similarity.items():
        if v:
            scores[k] = np.round(np.min(v) + scores_symmetry[k-1], 2)
        else:
            scores[k] = - np.inf
    values_rot = np.array([scores[i] for i in range(1, len(scores)+1)])
    indices = np.argsort(values_rot)[::-1] + 1
    for i in indices[:400]:
        print(i, values_rot[i-1])
    print("Charizard is ",values_rot[326])
    # np.sum(values_rot > 0.5)
    # values_rot[489]    

    for k, v in scores_features.items():
        if v:
            scores[k] = np.round(np.min(v) + scores_symmetry[k-1], 2)
        else:
            scores[k] = - np.inf
    values_feat = np.array([scores[i] for i in range(1, len(scores)+1)])
    np.argsort(values_feat)[::-1][:20] + 1
    # values_feat[346]
    final_scores = values_feat*values_rot
    # print(final_scores)
    final_scores = np.array([-np.inf if f == np.inf else f for f in final_scores])
    indices = np.argsort(final_scores)[::-1] + 1
    for i in indices[:100]:
        print(i, final_scores[i-1])
    print("Charizard is ", final_scores[326])
    print("Elephant is ", final_scores[3])
    print("Symmetric object is ", final_scores[282])
    print("Symmetric object is ", final_scores[18])
    np.savetxt("cfg/tools/data/final_scores",final_scores)
    # print(list(indices))
    # for k in indices:
    #     try:
    #         print(k, final_scores[k-1])
    #         img = plt.imread("pictures/allobj/obj" + str(k) + ".png")
    #         plt.imshow(img)
    #         plt.show()
    #     except:
    #         print("Image ", k, " not in objects")
