import time

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                         postprocessor)
        
        ##
        print("****************")
        print("saiful : this line is coming from /OpenOOD/openood/pipelines/test_ood_pipeline.py directory ")
        print("id_loader_dict['test'] :", id_loader_dict['test'])
        test_loader = id_loader_dict["test"]
        print("len(test_loader.dataset) :", len(test_loader.dataset) )
        test_features_dict = next(iter(test_loader))
        print("test_features_dict.keys()", test_features_dict.keys())
        test_features = test_features_dict["data"]
        test_labels = test_features_dict["label"]
        print("len(test_features):", len(test_features))
        print("len(test_labels):", len(test_labels))
        print("****************")
        ##
        print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        timer = time.time()
        evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)
