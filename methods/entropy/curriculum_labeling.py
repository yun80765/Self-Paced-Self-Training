from cgitb import enable
import os
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger
from utils.helpers import *
from utils.scheduler_ramps import *
from ..base import *
from .focal_loss import *
import math

# from https://github.com/pytorch/contrib = pip install torchcontrib
from torchcontrib.optim import SWA
import matplotlib.pyplot as plt

class Curriculum_Labeling(Train_Base):
    """
    Curriculum Labeling, method proposed by Cascante-Bonilla et. al. in https://arxiv.org/abs/2001.06001.
    """
    def __init__(self, args, model, model_optimizer):
        """
        Initialize the Curriculum Learning class with all methods and required variables
        This class use the model, optimizer, dataloaders and all the user parameters to train the CL algorithm proposed by Cascante-Bonilla et. al. in Curriculum Learning: (https://arxiv.org/abs/2001.06001)

        Args:
            args (dictionary): all user defined parameters with some pre-initialized objects (e.g., model, optimizer, dataloaders)
        """
        self.best_prec1 = 0

        ### list error and losses ###
        self.train_class_loss_list = []
        self.train_error_list = []
        self.train_lr_list = []
        self.val_class_loss_list = []
        self.val_error_list = []

        exp_dir = os.path.join(args.root_dir, '{}/{}/{}'.format(args.dataset, args.arch, args.add_name))
        prGreen('Results will be saved to this folder: {}'.format(exp_dir))

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        self.args = args
        self.args.exp_dir = exp_dir
        self.model = model
        self.model_optimizer = model_optimizer
        self.iteration = 0

        # add TF Logger
        self.train_logger = Logger(os.path.join(self.args.exp_dir, 'TFLogs/train'))
        self.val_logger = Logger(os.path.join(self.args.exp_dir, 'TFLogs/val'))

    # mixup code from: https://arxiv.org/pdf/1710.09412.pdf ==> https://github.com/facebookresearch/mixup-cifar10
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        """
        Data augmentation technique proposed by Zhang et. al. that consists on interpolating two samples and their corresponding labels (https://arxiv.org/pdf/1710.09412.pdf).

        Args:
            x: input batch
            y: target batch
            alpha (float, optional): ~ Beta(alpha, alpha). Defaults to 1.0.
            use_cuda (bool, optional): if cuda is available. Defaults to True.

        Returns:
            Mixed inputs, pairs of targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Loss function to compute when mixup is applied.

        Args:
            criterion: loss function (e.g. categorical crossentropy loss)
            pred: model output
            y_a: true y_a
            y_b: true y_a
            lam: lambda (interpolation ratio)

        Returns:
            loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    # end of mixup code from: https://arxiv.org/pdf/1710.09412.pdf ==> https://github.com/facebookresearch/mixup-cifar10

    def train_models(self, trainloader, unlabelledloader, unlabelled_sampler, indices_unlabelled, validloader, testloader, modelName, model, optimizer, train_logger, val_logger, num_classes = 10, hard_labeles_for_rotation = {}, init_epoch = 0):
        """
        Method to train the Curriculum Labeling method.
        When no pseudo labels are given: train_base, else: train_pseudo.

        Args:
            trainloader: labeled subset loader
            unlabelledloader: unlabeled subset loader
            unlabelled_sampler: unlabeled subset sampler - useful for pseudo-annotating the data retrieved from unlabelledloader
            indices_unlabelled: indices of the unlabeled set - useful for pseudo-annotating the data retrieved from unlabelledloader
            validloader: validation subset loader
            testloader: test subset loader (when debug is enabled)
            modelName: name given to the model for saving checkpoints
            model: model instance
            optimizer: predefined optimizer assigned to model
            train_logger: TensorBoard instance logger for the training process
            val_logger: TensorBoard instance logger for the validation process
            num_classes (int, optional): number of classes - bound to dataset and user defined available classes. Defaults to 10.
            hard_labeles_for_rotation (dict, optional): dictionary containing the pseudo annotated samples (sample_index: annotation). Defaults to {}.
            init_epoch (int, optional): initial epoch to start training - could vary when finetuning. Defaults to 0.
        """
        for epoch in range(init_epoch, self.args.epochs + (self.args.lr_rampdown_epochs-self.args.epochs)):
            start_time = time.time()
            if len(hard_labeles_for_rotation) > 0:
                hLabels = np.zeros((len(hard_labeles_for_rotation), self.args.num_classes))
                for i, k in enumerate(hard_labeles_for_rotation):
                    hLabels[i][hard_labeles_for_rotation[k]] = 1
                w = self.get_label_weights(hLabels)
                w = torch.FloatTensor(w/100).cuda()
                self.train_pseudo(unlabelledloader, unlabelled_sampler, indices_unlabelled, hard_labeles_for_rotation, model, optimizer, epoch, self.train_logger, modelName, weights = w, use_zca = self.args.use_zca)
            else:
                w = torch.FloatTensor(np.full(self.args.num_classes, 0.1)).cuda()
                self.train_base(trainloader, model, optimizer, epoch, self.train_logger, use_zca = self.args.use_zca)

            if self.args.swa:
                if epoch > self.args.swa_start and epoch%self.args.swa_freq == 0 :
                    optimizer.swap_swa_sgd()
                    if len(hard_labeles_for_rotation) > 0:
                        optimizer.bn_update(unlabelledloader, model, torch.device("cuda"))
                    else:
                        optimizer.bn_update(trainloader, model, torch.device("cuda"))
                    optimizer.swap_swa_sgd()

            print("--- training " + modelName + " epoch in %s seconds ---" % (time.time() - start_time))

            # evaluate, save best model and log results on console and TensorBoard logger
            self.evaluate_after_train(modelName, validloader, testloader, model, optimizer, epoch)
            # self.vis(self.acc_per_class)

    def update_args(self, args, model, model_optimizer, update_model=False):
        """
        Useful when the data subsets are updated outside this class.

        Args:
            args (dictionary): dict of parameters set by user or updated by external methods.
        """
        self.args = args
        if update_model:
            self.model = model
            self.model_optimizer = model_optimizer

    def train_iteration(self, iteration=0, image_indices_hard_label={}):
        """
        Train model. Resets the best precision 1 variable to 0 and calls the train_models function.
        Usually, when image_indices_hard_label is empty and iteration is 0, the model is trained using only the labeled subset.

        Args:
            iteration (int, optional): curriculum labeling iteration. Defaults to 0.
            image_indices_hard_label (dict, optional): dictionary of pseudo annotated samples. Defaults to {}.
        """
        self.best_prec1 = 0
        self.acc_per_class = torch.empty(0,10)
        self.train_models(self.args.trainloader, self.args.unlabelledloader, self.args.unlabelled_sampler, self.args.indices_unlabelled, self.args.validloader, self.args.testloader, '{}-Rotation'.format(iteration), \
                    self.model, self.model_optimizer, self.train_logger, self.val_logger, num_classes = self.args.num_classes, hard_labeles_for_rotation = image_indices_hard_label, init_epoch = 0)

    def do_iteration(self, iteration):
        """
        Executes steps 2-5 of the Curriculum Learning algorithm.

        2) Use trained model to get max scores of unlabeled data
        3) Compute threshold (check percentiles_holder parameter) based on max scores -> long tail distribution
        4) Pseudo-label
        5) Train next iteration

        Args:
            iteration (int): curriculum labeling iteration

        Returns:
            image_indices_hard_label: dictionary of pseudo annotated samples.
        """
        # sets mu: percentiles threshold
        percentiles_holder = 100 - (self.args.percentiles_holder) * iteration
        print (f'percentiles_holder: {percentiles_holder}')
        self.iteration = iteration
        #load best model
        best_features_checkpoint = self.args.exp_dir + '/{}-Rotation.best.ckpt'.format(iteration-1)
        print("=> loading pretrained checkpoint '{}'".format(best_features_checkpoint))
        checkpoint = torch.load(best_features_checkpoint)
        self.best_prec1 = checkpoint['best_prec1']
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained checkpoint '{}' (epoch {}, best_prec {})".format(best_features_checkpoint, checkpoint['epoch'], checkpoint['best_prec1']))

        # get the results of the unlabeled set evaluated on the trained model
        # whole_result = np.asarray(self.get_results_val(self.args.unlabelledloader, self.model))
        # print('unlabelledloader',len(self.args.unlabelledloader))
        # whole_result = torch.tensor(self.get_results_val(self.args.unlabelledloader, self.model))

        mc_results,mc_var,whole_result = self.mc_dropout_eval()
        # print('mc_results: ',mc_results)

        # where to hold top scores
        image_indices_hard_label = {}
        # image_indices_hard_label = torch.ones((len(whole_result),), dtype=torch.long, ) * -1
        max_values_per_image = {}
        # if algorithm in debug mode, sort and print the most confident values (top scores)
        # if self.args.debug:
        #     whole_result_copy = whole_result.reshape((-1)).copy()
         #     topScores = np.argpartition(whole_result_copy, -5)[-5:]
        #     topScores = topScores[np.argsort(whole_result_copy[topScores])]
        #     prRed('Top Scores: {}'.format(whole_result_copy[topScores]))

        # get top scores
        # for i in range (whole_result.shape[0]):
        #     hardlabel = whole_result[i].argmax()
        #     max_values.append(whole_result[i][hardlabel])
        max_values,max_idx = torch.max(whole_result,dim=-1)
        max_var = -(mc_var.gather(1, max_idx.view(-1,1)))
        print('max_var:',torch.min(max_var),torch.min(max_var))
        max_var = (max_var-max_var.min())/(max_var.max()-max_var.min())
        print('max_var:',torch.min(max_var),torch.min(max_var))
        # max_var = mc_var.gather(1, max_idx.view(-1,1))

        # set new threshold based on top scores
        if percentiles_holder < 0:
            percentiles_holder = 0
        # threshold = np.percentile(max_values, percentiles_holder)
        threshold = torch.quantile(max_values,percentiles_holder/100)
        var_threshold = torch.quantile(max_var,percentiles_holder/100)

        ############# compute threshold ################
        if self.args.classwise_curriculum:
            threshold_class = torch.zeros((self.args.num_classes,))
            var_threshold_class = torch.zeros((self.args.num_classes,))
            for i in range(self.args.num_classes):
                max_values_class = max_values[(max_idx==i).nonzero(as_tuple=True)]
                max_var_class = max_var[(max_idx==i).nonzero(as_tuple=True)]
                if len(max_values_class)>0:
                    threshold_class[i] = torch.quantile(max_values_class,percentiles_holder/100)
                    var_threshold_class[i] = torch.quantile(max_var_class,percentiles_holder/100)
                    idx_class = (max_idx==i).nonzero().squeeze()
                    idx_select = idx_class[max_values[idx_class].ge(threshold_class[i])]
                    var_select = idx_class[max_var[idx_class].ge(var_threshold_class[i]).squeeze()]
                    select = np.intersect1d(idx_select,var_select)
                    u_select = idx_select[(idx_select.view(1,-1) != var_select.view(-1, 1)).all(dim=0)]
                    select = idx_select
                    n = 0
                    # for j in u_select:
                    #     k = j
                    #     n+=1
                    #     for index, (input, target) in enumerate(self.args.unlabelledloader):
                    #         if k>=self.args.batch_size:
                    #             k -= self.args.batch_size
                    #         else:
                    #             if i!=target[k]:
                    #                 self.log_img_and_pseudolabels_and_uncertainty(input[k],i,target[k],max_values[j],max_var[j])
                    #             break
                    #         if n>200:
                    #             break

                    for s in select:
                        image_indices_hard_label[self.args.indices_unlabelled[s.item()]] = i
        else:
            ############## check max score against threshold to assign pseudo label #############
            print ('no classwise')
            sorted,indices = torch.sort(max_values,descending=True)
            n = len(max_values)
            num_per_class = torch.zeros((self.args.num_classes,))
            max_per_class = (n-(n*percentiles_holder/100))*2/self.args.num_classes
            num_pseudo = 0
            num_percentlines = n-(n*percentiles_holder/100)
            act_threshold = 0
            print('max_per_class:', max_per_class)
            for i in range(n):
                hardlabel = max_idx[i]
                # print(hardlabel.item())
                # if num_per_class[hardlabel] < max_per_class :
                if max_var[i] < 0.05 and max_values[i]>0.9:
                    image_indices_hard_label[self.args.indices_unlabelled[i]]=hardlabel.item()
                # num_per_class[hardlabel]+=1
                # num_pseudo+=1
                # act_threshold = sorted[i]
                if num_pseudo >= num_percentlines:
                    break





            # for i in range (whole_result.shape[0]):
            #     hardlabel = whole_result[i].argmax()
            #     # check top percent as threshold
            #     # if whole_result[i][hardlabel] >= threshold and BALD_acq[i] >= BALD_threshold:
            #     if whole_result[i][hardlabel] >= threshold:
            #     # if BALD_acq[i] >= BALD_threshold:
            #         image_indices_hard_label[self.args.indices_unlabelled[i]] = hardlabel.item()
            #     max_values_per_image[i] = [hardlabel, whole_result[i][hardlabel]]



            ############## compute classwise threshold ################
            threshold_class = torch.zeros((self.args.num_classes,))
            var_threshold_class = torch.zeros((self.args.num_classes,))
            for i in range(self.args.num_classes):
                max_values_class = max_values[(max_idx==i).nonzero(as_tuple=True)]
                max_var_class = max_var[(max_idx==i).nonzero(as_tuple=True)]
                # print('max_values_class: ',len(max_values_class))
                if len(max_values_class)>0:
                    var_threshold_class[i] = torch.quantile(max_var_class,percentiles_holder/100)
                    threshold_class[i] = torch.quantile(max_values_class,percentiles_holder/100)




        # prGreen ('Actual Threshold selected : {} '.format(act_threshold))
        prGreen ('Actual Threshold: {} - Percentile: {}'.format(threshold, percentiles_holder))
        prGreen ('Actual Threshold per class: {} - Percentile: {}'.format(threshold_class, percentiles_holder))
        prGreen ('Actual Uncertainty threshold: {} - Percentile: {}'.format(var_threshold, percentiles_holder))
        prGreen ('Actual Uncertainty threshold per class: {} - Percentile: {}'.format(var_threshold_class, percentiles_holder))
        prGreen ('[{} Rotation] | Total of hard-labeled images: {}'.format((iteration), len(image_indices_hard_label)))
        percentiles_holder -= self.args.percentiles_holder
        #add the labeled images
        for i, (_, target) in enumerate(self.args.trainloader):
            for t in range(len(target)):
                image_indices_hard_label[self.args.indices_train[(i*self.args.batch_size)+t]] = target[t].item()

        prGreen ('[{} Rotation] | Total of hard-labeled images + known labeled images: {}'.format(iteration, len(image_indices_hard_label)))
        pickle.dump(image_indices_hard_label, open(self.args.exp_dir + '/{}-Rotation_HardLabels.p'.format(iteration), "wb"))
        # pickle.dump(max_values_per_image, open(self.args.exp_dir + '/{}-Rotation_MaxValues.p'.format(iteration), "wb"))

        self.train_logger.scalar_summary('all/images', len(image_indices_hard_label), iteration)

        return image_indices_hard_label

    def train_base(self, trainloader, model, optimizer, epoch, train_logger, use_zca = True, weights = None):
        """
        Train using the labeled subset only.

        Args:
            trainloader: labeled subset data loader
            model: model instance
            optimizer: predefined optimizer assigned to model
            epoch: current epoch
            train_logger: instance to TensorBoard logs
            use_zca (bool, optional): use zca or not (for CIFAR10)
            weights (optional): class weights
        """
        class_criterion = nn.CrossEntropyLoss(weight = weights).cuda()
        meters = AverageMeterSet()
        model.train()
        end = time.time()

        for i, (input, target) in enumerate(trainloader):
            # measure data loading time
            meters.update('data_time', time.time() - end)
            if self.args.dataset == 'cifar10':
                if use_zca:
                    input = apply_zca(input, zca_mean, zca_components)

            if epoch <= self.args.epochs:
                lr = self.adjust_learning_rate(optimizer, epoch, i, len(trainloader))
            meters.update('lr', optimizer.param_groups[0]['lr'])
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda(non_blocking = True))

            # continue with standard training
            self.apply_train_common(model, class_criterion, optimizer, input_var, target_var, i, len(trainloader), epoch, meters, end)

        self.train_class_loss_list.append(meters['class_loss'].avg)
        self.train_error_list.append(meters['error1'].avg)
        self.train_lr_list.append(meters['lr'].avg)

        self.train_logger.scalar_summary('0-Rotation/train/loss', meters['class_loss'].avg, epoch)
        self.train_logger.scalar_summary('0-Rotation/train/prec1', meters['top1'].avg, epoch)
        self.train_logger.scalar_summary('0-Rotation/train/prec5', meters['top5'].avg, epoch)
        self.train_logger.scalar_summary('0-Rotation/train/lr', meters['lr'].avg, epoch)

        self.log_img_and_pseudolabels(input[0], target[0], self.train_logger)

    def train_pseudo(self, unlabelledloader, unlabelled_sampler, indices_unlabelled, hardLabeledResults, model, optimizer, epoch, train_logger, modelName, weights = None, use_zca = True):
        """
        Train using pseudo labeled samples along with the labeled subset.

        Args:
            unlabelledloader: unlabel subset data loader
            unlabelled_sampler: unlabel subset data sampler
            indices_unlabelled: indices of unlabeled set
            hardLabeledResults: dictionary containing the pseudo annotated samples (sample_index: annotation)
            model: model instance
            optimizer: predefined optimizer assigned to model
            epoch: current epoch
            train_logger: instance to TensorBoard logs
            use_zca (bool, optional): use zca or not (for CIFAR10)
            weights (optional): class weights
        """
        class_criterion = nn.CrossEntropyLoss(weight = weights).cuda()
        # class_criterion = nn.CrossEntropyLoss().cuda()
        # class_criterion = focal_loss(alpha=weights,gamma=0).cuda()
        meters = AverageMeterSet()
        model.train()
        end = time.time()

        for i, (input, _) in enumerate(unlabelledloader):
            # measure data loading time
            meters.update('data_time', time.time() - end)
            if i == 0:
                # get indexes to access the pseudo annotations
                unlabOrigIdx = unlabelled_sampler.getOriginalIndices()
            if self.args.dataset == 'cifar10':
                if use_zca:
                    input = apply_zca(input, zca_mean, zca_components)

            if epoch <= self.args.epochs:
                lr = self.adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
            meters.update('lr', optimizer.param_groups[0]['lr'])
            input_var = torch.autograd.Variable(input.cuda())

            # now assign the pseudo labels
            newTarget = torch.zeros(len(input), dtype=torch.long)
            for t in range(len(input)):
                # get the image index
                indexInFile = indices_unlabelled[unlabOrigIdx[(i*self.args.batch_size)+t].item()]
                # assign pseudo targets
                fakeLabel = torch.tensor(hardLabeledResults[indexInFile], dtype=torch.long)
                newTarget[t] = fakeLabel
            target_var = torch.autograd.Variable(newTarget.cuda(non_blocking = True))
            # continue with standard training
            self.apply_train_common(model, class_criterion, optimizer, input_var, target_var, i, len(unlabelledloader), epoch, meters, end)

        self.train_class_loss_list.append(meters['class_loss'].avg)
        self.train_error_list.append(meters['error1'].avg)
        self.train_lr_list.append(meters['lr'].avg)

        self.train_logger.scalar_summary('{}/train/loss'.format(modelName), meters['class_loss'].avg, epoch)
        self.train_logger.scalar_summary('{}/train/prec1'.format(modelName), meters['top1'].avg, epoch)
        self.train_logger.scalar_summary('{}/train/prec5'.format(modelName), meters['top5'].avg, epoch)
        self.train_logger.scalar_summary('{}/train/lr'.format(modelName), meters['lr'].avg, epoch)

        self.log_img_and_pseudolabels(input[0], target_var[0], self.train_logger)

    def get_results_val(self, eval_loader, model):
        """
        Get results from the current model evaluated on a subset.

        Args:
            eval_loader: data loader
            model: model instance

        Returns:
            array: contains the results of the evaluated subset (the output of the model after applying a softmax operation)
        """
        setResults = []
        model.eval()

        end = time.time()
        for i, (input, _) in enumerate(eval_loader):

            if self.args.dataset == 'cifar10':
                if self.args.use_zca:
                    input = apply_zca(input, zca_mean, zca_components)

            with torch.no_grad():
                input_var = torch.autograd.Variable(input.cuda())

            output1 = model(input_var)
            softmax1 = F.softmax(output1, dim=1)

            setResults.extend(softmax1.cpu().detach().numpy())

        return setResults

    def evaluate_all_iterations(self):
        iteration = 0
        val_best = 0
        test1_best = 0
        test5_best = 0
        iter_best = 0
        classwise_acc = torch.empty(0,self.args.num_classes)
        while self.args.percentiles_holder * iteration <= 100:
            print("=> loading pretrained checkpoint '{}'".format(self.args.exp_dir + '/{}-Rotation.best.ckpt'.format(iteration)))
            checkpoint = torch.load(self.args.exp_dir  + '/{}-Rotation.best.ckpt'.format(iteration))
            best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(self.args.exp_dir + '/{}-Rotation.best.ckpt'.format(iteration), checkpoint['epoch']))

            prGreen("Val: Evaluating the model:")
            top1_val, top5_val, loss_val,classwise_acc_val = self.validate(self.args.validloader, self.model, self.args.start_epoch)
            print ('=====================================================')
            prGreen("Test: Evaluating the model:")
            top1_test, top5_test, loss_test,classwise_acc_test = self.validate(self.args.testloader, self.model, self.args.start_epoch)
            classwise_acc = torch.vstack((classwise_acc,torch.tensor(classwise_acc_test).unsqueeze(0)))
            if top1_val > val_best:
                val_best = top1_val
                test1_best = top1_test
                test5_best = top5_test
                iter_best = iteration

            print('')
            iteration += 1

        self.vis(classwise_acc)
        prGreen('Final top-1 test accuracy: {}, top-5 test accuracy: {} || {}-th iteration'.format(test1_best, test5_best, iter_best))

    def mc_dropout_eval(self):
        self.enable_dropout(self.model)
        mc_predictions = torch.zeros(self.args.t_mc_dropout,len(self.args.indices_unlabelled),self.args.num_classes)
        for t in range(self.args.t_mc_dropout):
            setResults = []
            for i, (input, _) in enumerate(self.args.unlabelledloader):

                if self.args.dataset == 'cifar10':
                    if self.args.use_zca:
                        input = apply_zca(input, zca_mean, zca_components)

                with torch.no_grad():
                    input_var = torch.autograd.Variable(input.cuda())

                output1 = self.model(input_var)
                softmax1 = F.softmax(output1, dim=1)

                setResults.extend(softmax1.cpu().detach().numpy())
            mc_predictions[t] = torch.tensor(setResults)
        self.model.train()
        mc_var,mc_mean = torch.var_mean(mc_predictions,dim=0)
        return mc_predictions,mc_var,mc_mean


    def enable_dropout(self,model):
        """ Function to enable the dropout layers during test-time """
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def get_BALD_acquisition(self,mc_results):
        expected_entropy = - torch.mean(torch.sum(mc_results * torch.log(mc_results + 1e-10), dim=-1), dim=0)
        expected_p = torch.mean(mc_results, dim=0)
        entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-10), dim=-1)
        return (entropy_expected_p - expected_entropy)

    def vis(self,acc_per_class):
        print('acc_per_class: ',acc_per_class)
        plt.figure()
        x = acc_per_class.t()
        # print('acc_per_class:', x)
        for i in range(self.args.num_classes):
            plt.plot(x[i],label=i)
        plt.legend(loc='lower right')
        plt.title('acc per class')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(f'./figure/{self.args.add_name}_acc_per_class.png')


    def log_img_and_pseudolabels_and_uncertainty(self, input_img, pseudolabel, label, confidence, uncertainty):
        """
        Add image and label or pseudo-label to TensorBoard log for debugging purposes.

        Args:
            input_img: image
            pseudolabel: label
            logger: log reference
        """
        anns = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        img = input_img.clone()
        # print(input_img)
        # print(img)
        img = img.squeeze() # get rid of batch dim.
        if self.args.dataset == 'cifar10':
            # un-normalize pixel values.
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            _label = anns[pseudolabel]
            true_label = anns[label.item()]
            _orig_image = img.transpose(0,2).transpose(0,1).cpu().detach().numpy()

            plt.figure()
            plt.imshow(_orig_image)
            plt.title(_label)
            plt.savefig(f'./u_figure2/{self.iteration}_{_label}_{true_label}_{round(confidence.item(),3)}_{round(uncertainty.item(),3)}.png')
            print(f'./u_figure2/{self.iteration}_{_label}_{true_label}_{round(confidence.item(),3)}_{round(uncertainty.item(),3)}.png saved')
            # breakpoint()


