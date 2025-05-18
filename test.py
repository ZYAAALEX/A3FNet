from torch.utils.data import DataLoader
from dataset import Test_datasets
from model.A3Fnet import A3Fnet
from config_setting import setting_config
from tqdm import tqdm
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")
def test_one_epoch(test_loader, model, criterion, logger, config, path, test_data_name=None):
    model.eval()
    gt_list = []
    pred_list = []
    total_miou = 0.0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            img, msk, img_name = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            gt_pre, key_points, out = model(img)
            msk = msk.squeeze(1).cpu().detach().numpy()
            out = out.squeeze(1).cpu().detach().numpy()
            gt_list.append(msk)
            pred_list.append(out)
            y_pre = np.where(out>=config.threshold, 1, 0)
            y_true = np.where(msk>=0.5, 1, 0)
            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)
            total_miou += miou
            total += 1

            save_imgs(out,  img_name,config.work_dir + 'outputs\\' ,
                      config.datasets, config.threshold)

        total_miou = total_miou / total

        pred_list = np.array(pred_list).reshape(-1)
        gt_list = np.array(gt_list).reshape(-1)

        y_pre = np.where(pred_list>=0.5, 1, 0)
        y_true = np.where(gt_list>=0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        print(confusion)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0

        log_info = f'test of best model, miou: {total_miou}, miou::{miou},f1_or_dsc: { dsc}, recall:{recall}, precision{precision}'
        print(log_info)
        logger.info(log_info)

def main(config):
    config.work_dir = '/A3FNet/'
    log_dir = os.getcwd()
    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')

    test_dataset =Test_datasets(config.data_path, config)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=False)
    
    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'A3Fnet':
        model = A3Fnet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'], 
                        c_list=model_cfg['c_list'], 
                        )
    else: raise Exception('network in not right!')
    model = model.cuda()

    input_path = ''

    if os.path.exists(input_path):
        print('#----------Testing----------#')
        best_weight = torch.load(input_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight, strict=False)
        test_one_epoch(
                test_loader,
                model,
                config.criterion,
                logger,
                config,
                path = ''
        )

if __name__ == '__main__':
    config = setting_config
    main(config)