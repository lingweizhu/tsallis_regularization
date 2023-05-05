
def extract_from_single_run(file, key, label=None, before_step=None, testset=None, xlim=[]):
    with open(file, "r") as f:
        content = f.readlines()
        # print(file)
    returns = []
    
    for num, l in enumerate(content):
        if "|" in l:
            info = l.split("|")[1].strip()
            i_list = info.split(" ")
            # print('i_list: ', i_list)
            if key == "learning_rate" and "learning_rate:" in i_list:
                returns = i_list[1]
                return returns
            if key == "tau" and "tau:" in i_list:
                returns = i_list[1]
                return returns
            elif key == "id" and "id:" in i_list:
                returns = i_list[1]
                return returns
            elif key == "train_loss" and "training" in i_list and "loss" in i_list:
                returns.append(float(i_list[i_list.index("training")+2].split("/")[0].strip()))
            elif key == "test_loss" and "test" in i_list and "loss" in i_list:
                returns.append(float(i_list[i_list.index("test")+2].split("/")[0].strip()))
            if "EVAL:" == i_list[0] or "total" == i_list[0] or "TRAIN" == i_list[0] or "TEST" == i_list[0] or "Normalized" == i_list[0]:
                if key == "train_return" and "returns" in i_list and "TRAIN" == i_list[0]:
                    returns.append(float(i_list[i_list.index("returns") + 1].split("/")[0].strip()))  # mean
                elif key == "test_return" and "returns" in i_list and "TEST" == i_list[0]:
                    returns.append(float(i_list[i_list.index("returns") + 1].split("/")[0].strip()))  # mean
                    # returns.append(float(i_list[i_list.index("returns")+1].split("/")[1].strip())) # median
                    # returns.append(float(i_list[i_list.index("returns")+1].split("/")[2].strip())) # min
                    # returns.append(float(i_list[i_list.index("returns")+1].split("/")[3].strip())) # max
                elif key == "normalized_return" and "returns" in i_list and "Normalized" == i_list[0]:
                    returns.append(float(i_list[i_list.index("returns") + 1].split("/")[0].strip()))  # mean

    # Sanity Check
    if not isinstance(returns, int):
        if len(returns) in [0]:  # , 1] :
            print('Empty returns when extracting: {} '.format(key), returns)
            print('File Name: ', file)
            
    if xlim!=[]:
        return returns[xlim[0]: xlim[1]]
    else:
        return returns

