'''

@author: "epsilonyuan@gmail.com"
'''
from injector import Module, provides, Key, inject, ClassProvider
from pymfr.model.searcher import SupportNodeSearcher1D, RawSupportNodeSearcher, SupportNodeSearcher, \
    VisibleSupportNodeSearcher2D, SegmentSearcher, KDTreeSegmentSearcher, KDTreeNodeSearcher

CoreSupportNodeSearcher = Key('core_support_node_searcher')

class OneDSupportNodesSearcherModule(Module):
    @provides(SupportNodeSearcher)
    @inject(support_node_searcher=CoreSupportNodeSearcher)
    def support_node_searcher(self, support_node_searcher):
        return SupportNodeSearcher1D(support_node_searcher)
    
    @provides(CoreSupportNodeSearcher)
    def core_supprt_node_searcher(self):
        return RawSupportNodeSearcher()

class TwoDVisibleSupportNodeSearcherModule(Module):
    
    def __init__(self, core_node_searcher_cls=None, segment_searcher_cls=None):
        self.core_node_searcher_cls = (core_node_searcher_cls if core_node_searcher_cls is not None 
                                  else KDTreeNodeSearcher)
        self.segment_searcher_cls = (segment_searcher_cls if segment_searcher_cls is not None 
                                     else KDTreeSegmentSearcher)
        super().__init__()
    
    @provides(SupportNodeSearcher)
    @inject(
            support_node_searcher=CoreSupportNodeSearcher,
            segment_searcher=SegmentSearcher,
            )
    def support_node_searcher(self, **kwargs):
        return VisibleSupportNodeSearcher2D(**kwargs)
    
    
    def configure(self, binder):
        binder.bind(SegmentSearcher, to=ClassProvider(self.segment_searcher_cls))
        binder.bind(CoreSupportNodeSearcher, to=ClassProvider(self.core_node_searcher_cls))
