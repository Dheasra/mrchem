#include <iostream>
#include "SerialTree.h"
#include "MWNode.h"
#include "MWTree.h"
#include "FunctionTree.h"
#include "FunctionNode.h"
#include "ProjectedNode.h"
#include "GenNode.h"
#include "MathUtils.h"
#include "Timer.h"
#include "parallel.h"

using namespace std;
using namespace Eigen;

/** SerialTree class constructor.
  * Allocate the root FunctionNodes and fill in the empty slots of rootBox.
  * Initializes rootNodes to represent the zero function and allocate their nodes. 
  * NOTES:
  * Serial trees are made of projected nodes, and include gennodes and loose nodes separately.
  * All created (using class creator) Projected nodes or GenNodes are loose nodes. 
  * Loose nodes have their coeff in serial Tree, but not the node part. 
  * Projected nodes and GenNodes that are created by their creator, are detroyed by destructor ~ProjectedNode and ~GenNode. 
  * Serial tree nodes are not using the destructors, but explicitely call to deallocNodes or deallocGenNodes
  * Gen nodes and loose nodes are not counted with MWTree->[in/de]crementNodeCount()
*/
template<int D>
SerialTree<D>::SerialTree(FunctionTree<D> *tree, int max_nodes)
        : maxNodes(max_nodes),
          maxGenNodes(max_nodes),
          nNodes(0),
          nGenNodes(0),
          nNodesCoeff(0),
          nGenNodesCoeff(0),
          tree_p(tree),
          lastNode(0),
          lastGenNode(0) {

    //Size for GenNodes chunks. ProjectedNodes will be 8 times larger
    int sizePerChunk = 1024*1024;// 1 MB small for no waisting place, but large enough so that latency and overhead work is negligible

    this->sizeGenNodeCoeff = this->tree_p->getKp1_d();//One block
    this->sizeNodeCoeff = (1<<D)*this->sizeGenNodeCoeff;//TDim  blocks
    println(10, "SizeNode Coeff (kB) " << this->sizeNodeCoeff*sizeof(double)/1024);
    println(10, "SizeGenNode Coeff (kB) " << this->sizeGenNodeCoeff*sizeof(double)/1024);

    this->maxNodesPerChunk = sizePerChunk/this->sizeGenNodeCoeff;
    println(10, " max_nodes = " << max_nodes << ", nodes per chunk = " << this->maxNodesPerChunk);

    //indicate occupation of nodes
    this->nodeStackStatus = new int[this->maxNodes + 1];
    this->genNodeStackStatus = new int[this->maxGenNodes + 1];

    this->lastNode = (ProjectedNode<D>*) this->sNodes;//position of last allocated node
    this->lastGenNode = this->sGenNodes;//position of last allocated Gen node

    //initialize stacks
    for (int i = 0; i < maxNodes; i++) {
        this->nodeStackStatus[i] = 0;//0=unoccupied
    }
    this->nodeStackStatus[maxNodes] = -1;//=unavailable

    //initialize stacks
    for (int i = 0; i < this->maxGenNodes; i++) {
        this->genNodeStackStatus[i] = 0;//0=unoccupied
    }
    this->genNodeStackStatus[this->maxGenNodes] = -1;//=unavailable

#ifdef HAVE_OPENMP
    omp_init_lock(&Stree_lock);
#endif

    //make virtual table pointers
    ProjectedNode<D>* tmpNode = new ProjectedNode<D>();
    this->cvptr_ProjectedNode =  *(char**)(tmpNode);
    delete tmpNode;

    GenNode<D>* tmpGenNode = new GenNode<D>();
    this->cvptr_GenNode =  *(char**)(tmpGenNode);
    delete tmpGenNode;

}

template<int D>
void SerialTree<D>::allocRoots(FunctionTree<D> &tree) {
    int sIx;
    double *coefs_p;
    //reserve place for nRoots
    int nRoots = tree.getRootBox().size();
    ProjectedNode<D> *root_p = this->allocNodes(nRoots, &sIx, &coefs_p);

    MWNode<D> **roots = tree.getRootBox().getNodes();
    for (int rIdx = 0; rIdx < nRoots; rIdx++) {
        roots[rIdx] = root_p;

        *(char**)(root_p) = this->cvptr_ProjectedNode;

        root_p->tree = &tree;
        root_p->parent = 0;
        for (int i = 0; i < (1 << D); i++) {
            root_p->children[i] = 0;
        }

        root_p->nodeIndex = tree.getRootBox().getNodeIndex(rIdx);
        root_p->hilbertPath = HilbertPath<D>();

        root_p->n_coefs = this->sizeNodeCoeff;
        root_p->coefs = coefs_p;

        root_p->serialIx = sIx;
        root_p->parentSerialIx = -1;//to indicate rootnode
        root_p->childSerialIx = -1;

        root_p->status = 0;

        root_p->clearNorms();
        root_p->setIsLeafNode();
        root_p->setIsAllocated();
        root_p->clearHasCoefs();
        root_p->setIsEndNode();
        root_p->setHasWCoefs();//default until known
        root_p->setIsRootNode();

        tree.incrementNodeCount(root_p->getScale());

#ifdef OPENMP
        omp_init_lock(&(root_p->node_lock));
#endif

        sIx++;
        root_p++;
        coefs_p += this->sizeNodeCoeff;
    }
}

template<int D>
void SerialTree<D>::allocChildren(FunctionNode<D> &parent) {
    int sIx;
    double *coefs_p;
    //NB: serial tree MUST generate all children consecutively
    //all children must be generated at once if several threads are active
    int nChildren = parent.getTDim();
    ProjectedNode<D> *child_p = this->allocNodes(nChildren, &sIx, &coefs_p);

    //position of first child
    parent.childSerialIx = sIx;
    for (int cIdx = 0; cIdx < nChildren; cIdx++) {
        parent.children[cIdx] = child_p;

        *(char**)(child_p) = this->cvptr_ProjectedNode;

        child_p->tree = parent.tree;
        child_p->parent = &parent;
        for (int i = 0; i < child_p->getTDim(); i++) {
            child_p->children[i] = 0;
        }

        child_p->nodeIndex = NodeIndex<D>(parent.getNodeIndex(), cIdx);
        child_p->hilbertPath = HilbertPath<D>(parent.getHilbertPath(), cIdx);

        child_p->n_coefs = this->sizeNodeCoeff;
        child_p->coefs = coefs_p;

        child_p->serialIx = sIx;
        child_p->parentSerialIx = parent.serialIx;
        child_p->childSerialIx = -1;

        child_p->status = 0;

        child_p->clearNorms();
        child_p->setIsLeafNode();
        child_p->setIsAllocated();
        child_p->clearHasCoefs();
        child_p->setIsEndNode();
        child_p->setHasWCoefs();

        child_p->tree->incrementNodeCount(child_p->getScale());

#ifdef OPENMP
        omp_init_lock(&child_p->node_lock);
#endif

        sIx++;
        child_p++;
        coefs_p += this->sizeNodeCoeff;
    }
}

template<int D>
void SerialTree<D>::allocGenChildren(FunctionNode<D> &parent) {
    int sIx;
    double *coefs_p;
    //NB: serial tree MUST generate all children consecutively
    //all children must be generated at once if several threads are active
    int nChildren = parent.getTDim();
    GenNode<D>* child_p = this->allocGenNodes(nChildren, &sIx, &coefs_p);

    //position of first child
    parent.childSerialIx = sIx;//not used fro Gennodes?
    for (int cIdx = 0; cIdx < nChildren; cIdx++) {
        parent.children[cIdx] = child_p;

        *(char**)(child_p) = this->cvptr_GenNode;

	child_p->tree = parent.tree;
	child_p->parent = &parent;
	for (int i = 0; i < child_p->getTDim(); i++) {
	    child_p->children[i] = 0;
	}

	child_p->nodeIndex = NodeIndex<D>(parent.getNodeIndex(), cIdx);
	child_p->hilbertPath = HilbertPath<D>(parent.getHilbertPath(), cIdx);

	child_p->n_coefs = this->sizeGenNodeCoeff;
	child_p->coefs = coefs_p;

	child_p->serialIx = sIx;
	child_p->parentSerialIx = parent.serialIx;
	child_p->childSerialIx = -1;

	child_p->status = 0;

        child_p->clearNorms();
	child_p->setIsLeafNode();
	child_p->setIsAllocated();
	child_p->clearHasCoefs();
	child_p->clearHasWCoefs();
	child_p->setIsGenNode();

	child_p->tree->incrementGenNodeCount();

#ifdef OPENMP
	omp_init_lock(&child_p->node_lock);
#endif

        sIx++;
        child_p++;
        coefs_p += this->sizeGenNodeCoeff;
    }
}

/** Overwrite all pointers defined in the tree.
  * Necessary after sending the tree 
  * could be optimized. Should reset other counters? (GenNodes...) */
template<int D>
void SerialTree<D>::rewritePointers(int nChunks){
    NOT_IMPLEMENTED_ABORT;
    /*
  int depthMax = 100;
  MWNode<D>* stack[depthMax*8];
  int slen = 0, counter = 0;

  this->nGenNodes = 0;
  this->nGenNodesCoeff = -1;
  this->nLooseNodesCoeff = 0;

  //reinitialize stacks
  for (int i = 0; i < this->maxNodes; i++) {
        this->nodeStackStatus[i] = 0;
  }

  for (int i = 0; i < this->maxGenNodes; i++) {
        this->genNodeStackStatus[i] = 0;//0=unoccupied
  }
  this->genNodeStackStatus[this->maxGenNodes] = -1;//=unavailable

  for (int i = 0; i < this->maxLooseNodesCoeff; i++) {
    this->looseCoeffStackStatus[i] = 0;//0=unoccupied
  }
  this->looseCoeffStackStatus[this->maxLooseNodesCoeff]=-1;//-1=unavailable

  this->getTree()->nNodes = 0;
  this->getTree()->nodesAtDepth.clear();
  this->getTree()->squareNorm = 0.0;

  for(int ichunk = 0 ; ichunk < nChunks; ichunk++){
    for(int inode = 0 ; inode < this->maxNodesPerChunk; inode++){
      ProjectedNode<D>* node = (this->nodeChunks[ichunk]) + inode;
      if (node->SNodeIx >= 0) {
	  //node is part of tree, should be processed
	  this->getTree()->incrementNodeCount(node->getScale());
	  if (node->isEndNode()) this->getTree()->squareNorm += node->getSquareNorm();
	  
	  //normally (intel) the virtual table does not change, but we overwrite anyway
	  *(char**)(Node) = this->cvptr_ProjectedNode;
	  
	  Node->tree = this->getTree();

	  //"adress" of coefs is the same as node, but in another array
	  Node->coefs = this->NodeCoeffChunks[ichunk]+ inode*this->sizeNodeCoeff;
	  
	  //adress of parent and children must be corrected
	  //can be on a different chunks
	  if(Node->parentSNodeIx>=0){
	    int n_ichunk = Node->parentSNodeIx/this->maxNodesPerChunk;
	    int n_inode = Node->parentSNodeIx%this->maxNodesPerChunk;
	    Node->parent = this->NodeChunks[n_ichunk] + n_inode;
	  }else{Node->parent = 0;}
	    
	  for (int i = 0; i < Node->getNChildren(); i++) {
	    int n_ichunk = (Node->childSNodeIx+i)/this->maxNodesPerChunk;
	    int n_inode = (Node->childSNodeIx+i)%this->maxNodesPerChunk;
	    Node->children[i] = this->NodeChunks[n_ichunk] + n_inode;
	  }
	  this->NodeStackStatus[Node->SNodeIx] = 1;//occupied
#ifdef OPENMP
	  omp_init_lock(&(Node->node_lock));
#endif
	}

    }
  }

  //update other MWTree data
  FunctionTree<D>* Tree = static_cast<FunctionTree<D>*> (this->mwTree_p);

  NodeBox<D> &rBox = Tree->getRootBox();
  MWNode<D> **roots = rBox.getNodes();

  for (int rIdx = 0; rIdx < rBox.size(); rIdx++) {
    roots[rIdx] = (this->NodeChunks[0]) + rIdx;//adress of roots are at start of NodeChunks[0] array
  }

  this->getTree()->resetEndNodeTable();

*/
}

/** Make 8 children nodes with scaling coefficients from parent
 * Does not put 0 on wavelets
 */
template<int D>
void SerialTree<D>::GenS_nodes(MWNode<D>* Node){
    NOT_IMPLEMENTED_ABORT;
/*
  bool ReadOnlyScalingCoeff=true;

  //  if(Node->isGenNode())ReadOnlyScalingCoeff=true;
  if(Node->hasWCoefs())ReadOnlyScalingCoeff=false;
  double* cA;

  Node->genChildren();//will make children and allocate coeffs, but without setting values for coeffs.


  double* coeffin  = Node->coefs;
  double* coeffout = Node->children[0]->coefs;

  int Children_Stride = this->sizeGenNodeCoeff;
  S_mwTransform(coeffin, coeffout, ReadOnlyScalingCoeff, Children_Stride);
*/
}


/** Make children scaling coefficients from parent
 * Other node info are not used/set
 * coeff_in are not modified.
 * The output is written directly into the 8 children scaling coefficients. 
 * NB: ASSUMES that the children coefficients are separated by Children_Stride!
 */
template<int D>
void SerialTree<D>::S_mwTransform(double* coeff_in, double* coeff_out, bool readOnlyScaling, int stride, bool b_overwrite) {
    int operation = Reconstruction;
    int kp1 = this->getTree()->getKp1();
    int kp1_d = this->getTree()->getKp1_d();
    int tDim = (1<<D);
    int kp1_dm1 = MathUtils::ipow(kp1, D - 1);
    const MWFilter &filter = this->getTree()->getMRA().getFilter();
    double overwrite = 0.0;
    double *tmp;
    double tmpcoeff[kp1_d*tDim];
    double tmpcoeff2[kp1_d*tDim];
    int ftlim=tDim;
    int ftlim2=tDim;
    int ftlim3=tDim;
    if(readOnlyScaling){
        ftlim=1;
        ftlim2=2;
        ftlim3=4;
        //NB: Careful: tmpcoeff tmpcoeff2 are not initialized to zero
        //must not read these unitialized values!
    }

    overwrite = 0.0;
    int i = 0;
    int mask = 1;
    for (int gt = 0; gt < tDim; gt++) {
        double *out = tmpcoeff + gt * kp1_d;
        for (int ft = 0; ft < ftlim; ft++) {
            // Operate in direction i only if the bits along other
            // directions are identical. The bit of the direction we
            // operate on determines the appropriate filter/operator
            if ((gt | mask) == (ft | mask)) {
	        double *in = coeff_in + ft * kp1_d;
	        int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
	        const MatrixXd &oper = filter.getSubFilter(filter_index, operation);

	        MathUtils::applyFilter(out, in, oper, kp1, kp1_dm1, overwrite);
	        overwrite = 1.0;
            }
        }
        overwrite = 0.0;
    }
    if (D>1) {
        i++;
        mask = 2;//1 << i;
        for (int gt = 0; gt < tDim; gt++) {
            double *out = tmpcoeff2 + gt * kp1_d;
            for (int ft = 0; ft < ftlim2; ft++) {
                // Operate in direction i only if the bits along other
                // directions are identical. The bit of the direction we
                // operate on determines the appropriate filter/operator
	        if ((gt | mask) == (ft | mask)) {
	            double *in = tmpcoeff + ft * kp1_d;
	            int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
	            const MatrixXd &oper = filter.getSubFilter(filter_index, operation);
	  
	            MathUtils::applyFilter(out, in, oper, kp1, kp1_dm1, overwrite);
	            overwrite = 1.0;
	        }
            }
            overwrite = 0.0;
        }
    }
    if (D>2) {
        overwrite = 1.0;
        if(b_overwrite) overwrite = 0.0;
        i++;
        mask = 4;//1 << i;
        for (int gt = 0; gt < tDim; gt++) {
            double *out = coeff_out + gt * stride;//write right into children
            for (int ft = 0; ft < ftlim3; ft++) {
	        // Operate in direction i only if the bits along other
	        // directions are identical. The bit of the direction we
	        // operate on determines the appropriate filter/operator
                if ((gt | mask) == (ft | mask)) {
	            double *in = tmpcoeff2 + ft * kp1_d;
	            int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
	            const MatrixXd &oper = filter.getSubFilter(filter_index, operation);

	            MathUtils::applyFilter(out, in, oper, kp1, kp1_dm1, overwrite);
	            overwrite = 1.0;
                }
            }
            overwrite = 1.0;
            if(b_overwrite) overwrite = 0.0;
        }
    }

    if (D>3) MSG_FATAL("D>3 NOT IMPLEMENTED for S_mwtransform");

    if (D<3) {
        double *out;
        if(D==1)out=tmpcoeff;
        if(D==2)out=tmpcoeff2;
        if(b_overwrite){
            for (int j = 0; j < tDim; j++){ 
	        for (int i = 0; i < kp1_d; i++){ 
	            coeff_out[i+j*stride] = out[i+j*kp1_d];
	        }
            }
        }else{
            for (int j = 0; j < tDim; j++){ 
	        for (int i = 0; i < kp1_d; i++){ 
	            coeff_out[i+j*stride]+=out[i+j*kp1_d];
	        }
            }
        }
    }
}

/** Make parent from children scaling coefficients
 * Other node info are not used/set
 * coeff_in are not modified.
 * The output is read directly from the 8 children scaling coefficients. 
 * NB: ASSUMES that the children coefficients are separated by Children_Stride!
 */
template<int D>
void SerialTree<D>::S_mwTransformBack(double* coeff_in, double* coeff_out, int stride) {
    NOT_IMPLEMENTED_ABORT;
}

template<>
void SerialTree<3>::S_mwTransformBack(double* coeff_in, double* coeff_out, int stride) {
  int operation = Compression;
  int kp1 = this->getTree()->getKp1();
  int kp1_d = this->getTree()->getKp1_d();
  int tDim = 8;
  int kp1_dm1 = MathUtils::ipow(kp1, 2);
  const MWFilter &filter = this->getTree()->getMRA().getFilter();
  double overwrite = 0.0;
  double tmpcoeff[kp1_d*tDim];

  int ftlim = tDim;
  int ftlim2 = tDim;
  int ftlim3 = tDim;

  int i = 0;
  int mask = 1;
  for (int gt = 0; gt < tDim; gt++) {
        double *out = coeff_out + gt * kp1_d;
        for (int ft = 0; ft < ftlim; ft++) {
            // Operate in direction i only if the bits along other
            // directions are identical. The bit of the direction we
            // operate on determines the appropriate filter/operator
            if ((gt | mask) == (ft | mask)) {
	        double *in = coeff_in + ft * stride;
	        int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
	        const MatrixXd &oper = filter.getSubFilter(filter_index, operation);

	        MathUtils::applyFilter(out, in, oper, kp1, kp1_dm1, overwrite);
	        overwrite = 1.0;
            }
        }
        overwrite = 0.0;
    }
    i++;
    mask = 2;//1 << i;
    for (int gt = 0; gt < tDim; gt++) {
        double *out = tmpcoeff + gt * kp1_d;
        for (int ft = 0; ft < ftlim2; ft++) {
            // Operate in direction i only if the bits along other
            // directions are identical. The bit of the direction we
            // operate on determines the appropriate filter/operator
            if ((gt | mask) == (ft | mask)) {
	        double *in = coeff_out + ft * kp1_d;
	        int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
	        const MatrixXd &oper = filter.getSubFilter(filter_index, operation);

	        MathUtils::applyFilter(out, in, oper, kp1, kp1_dm1, overwrite);
	        overwrite = 1.0;
            }
        }
        overwrite = 0.0;
    }
    i++;
    mask = 4;//1 << i;
    for (int gt = 0; gt < tDim; gt++) {
        double *out = coeff_out + gt * kp1_d;
        //double *out = coeff_out + gt * N_coeff;
        for (int ft = 0; ft < ftlim3; ft++) {
            // Operate in direction i only if the bits along other
            // directions are identical. The bit of the direction we
            // operate on determines the appropriate filter/operator
            if ((gt | mask) == (ft | mask)) {
	        double *in = tmpcoeff + ft * kp1_d;
	        int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
	        const MatrixXd &oper = filter.getSubFilter(filter_index, operation);

	        MathUtils::applyFilter(out, in, oper, kp1, kp1_dm1, overwrite);
	        overwrite = 1.0;
            }
        }
        overwrite = 0.0;
    }
}


/** Allocating a Projected Serial (root) node.
  *
  * This routine creates an empty ProjectedNode node
  * with the appropriate translation */
template<int D>
ProjectedNode<D>* SerialTree<D>::createSnode(const NodeIndex<D> & nIdx) {
    NOT_IMPLEMENTED_ABORT;
/*

  int NodeIx;
  double *coefs_p;
  ProjectedNode<D>* newNode=this->allocNodes(1, &NodeIx, &coefs_p);

  *(char**)(newNode) = this->cvptr_ProjectedNode;
  newNode->SNodeIx = NodeIx;
  newNode->tree = this->mwTree_p;
  newNode->parent = 0;
  newNode->parentSNodeIx = -1;//to indicate rootnode
  newNode->nodeIndex = nIdx;
  newNode->hilbertPath = HilbertPath<D>();
  newNode->squareNorm = -1.0;
  newNode->status = 0;
  newNode->clearNorms();
  newNode->childSNodeIx = -1;
  for (int i = 0; i < (1 << D); i++) {
    newNode->children[i] = 0;
  }
  newNode->setIsLeafNode();
  newNode->coefs = coefs_p;
  newNode->n_coefs = this->sizeNodeCoeff;
  newNode->setIsAllocated();
  newNode->clearHasCoefs();
  
  newNode->tree->incrementNodeCount(newNode->getScale());
  newNode->setIsEndNode();
  newNode->setHasWCoefs();//default until known
  
#ifdef OPENMP
  omp_init_lock(&(newNode->node_lock));
#endif

  return newNode;
*/
}


//return pointer to the last active node or NULL if failed
template<int D>
ProjectedNode<D>* SerialTree<D>::allocNodes(int nAlloc, int *serialIx, double **coefs_p) {
    *serialIx = this->nNodes;
    int chunkIx = *serialIx%(this->maxNodesPerChunk);

    if (chunkIx == 0 or chunkIx+nAlloc > this->maxNodesPerChunk ) {
        //start on new chunk
        if (this->nNodes+nAlloc >= this->maxNodes){
            MSG_FATAL("maxNodes exceeded " << this->maxNodes);
        }

        //we want nodes allocated simultaneously to be allocated on the same pice.
        //possibly jump over the last nodes from the old chunk
        this->nNodes = this->maxNodesPerChunk*((this->nNodes+nAlloc-1)/this->maxNodesPerChunk);//start of next chunk

        int chunk = this->nNodes/this->maxNodesPerChunk;//find the right chunk

        //careful: nodeChunks.size() is an unsigned int
        if (chunk+1 > this->nodeChunks.size()){
	    //need to allocate new chunk
	    this->sNodes = (ProjectedNode<D>*) new char[this->maxNodesPerChunk*sizeof(ProjectedNode<D>)];
	    this->nodeChunks.push_back(this->sNodes);
            this->sNodesCoeff = new double[this->sizeNodeCoeff*this->maxNodesPerChunk];
            this->nodeCoeffChunks.push_back(this->sNodesCoeff);
	    if (chunk%100==99 and D==3) println(10,endl<<" number of nodes "<<this->nNodes <<",number of Nodechunks now " << this->nodeChunks.size()<<", total size coeff  (MB) "<<(this->nNodes/1024) * this->sizeNodeCoeff/128);
        }
        this->lastNode = this->nodeChunks[chunk] + this->nNodes%(this->maxNodesPerChunk);
        *serialIx = this->nNodes;
        chunkIx = *serialIx%(this->maxNodesPerChunk);
    }
    assert((this->nNodes+nAlloc-1)/this->maxNodesPerChunk < this->nodeChunks.size());

    ProjectedNode<D> *newNode  = this->lastNode;
    ProjectedNode<D> *newNode_cp  = newNode;
    *coefs_p = this->sNodesCoeff + chunkIx*this->sizeNodeCoeff;
 
    for (int i = 0; i < nAlloc; i++) {
        if (this->nodeStackStatus[*serialIx+i] != 0)
	    println(0, *serialIx+i<<" NodeStackStatus: not available " << this->nodeStackStatus[*serialIx+i]);
        this->nodeStackStatus[*serialIx+i] = 1;
        newNode_cp++;
    }
    this->nNodes += nAlloc;
    this->lastNode += nAlloc;

    return newNode;
}

template<int D>
void SerialTree<D>::deallocNodes(int serialIx) {
    if (this->nNodes < 0) {
        println(0, "minNodes exceeded " << this->nNodes);
        this->nNodes++;
    }
    this->nodeStackStatus[serialIx] = 0;//mark as available
    if (serialIx == this->nNodes-1) {//top of stack
        int topStack = this->nNodes;
        while (this->nodeStackStatus[topStack-1] == 0){
            topStack--;
            if (topStack < 1) break;
        }
        this->nNodes = topStack;//move top of stack
        //has to redefine lastNode
        int chunk = this->nNodes/this->maxNodesPerChunk;//find the right chunk
        this->lastNode = this->nodeChunks[chunk] + this->nNodes%(this->maxNodesPerChunk);
    }
}

//return pointer to the last active node or NULL if failed
template<int D>
GenNode<D>* SerialTree<D>::allocGenNodes(int nAlloc, int *serialIx, double **coefs_p) {
    omp_set_lock(&Stree_lock);
    *serialIx = this->nGenNodes;
    int chunkIx = *serialIx%(this->maxNodesPerChunk);
  
    //Not necessarily wrong, but new:
    assert(nAlloc == (1<<D));

    if(chunkIx == 0  or  chunkIx+nAlloc > this->maxNodesPerChunk ){
        //start on new chunk
        if (this->nGenNodes+nAlloc >= this->maxGenNodes){
	    MSG_FATAL("maxNodes exceeded " << this->maxGenNodes);
        } 

        //we want nodes allocated simultaneously to be allocated on the same chunk.
        //possibly jump over the last nodes from the old chunk
        this->nGenNodes=this->maxNodesPerChunk*((this->nGenNodes+nAlloc-1)/this->maxNodesPerChunk);//start of next chunk

        int chunk = this->nGenNodes/this->maxNodesPerChunk;//find the right chunk

        //careful: nodeChunks.size() is an unsigned int
        if(chunk+1 > this->genNodeChunks.size()){
	    //need to allocate new chunk
	    this->sGenNodes = (GenNode<D>*) new char[this->maxNodesPerChunk*sizeof(GenNode<D>)];
	    this->genNodeChunks.push_back(this->sGenNodes);
	    this->sGenNodesCoeff = new double[this->sizeGenNodeCoeff*this->maxNodesPerChunk];
	    genNodeCoeffChunks.push_back(this->sGenNodesCoeff);
	    if(chunk%100==99 and D==3)println(10,endl<<" number of GenNodes "<<this->nGenNodes <<",number of GenNodechunks now " << this->genNodeChunks.size()<<", total size coeff  (MB) "<<(this->nGenNodes/1024) * this->sizeGenNodeCoeff/128);
        }
        this->lastGenNode = this->genNodeChunks[chunk] + this->nGenNodes%(this->maxNodesPerChunk);
        *serialIx = this->nGenNodes; 
        chunkIx = *serialIx%(this->maxNodesPerChunk);
    }
    assert((this->nGenNodes+nAlloc-1)/this->maxNodesPerChunk < this->genNodeChunks.size());

    GenNode<D>* newNode  = this->lastGenNode;
    GenNode<D>* newNode_cp  = newNode;
    *coefs_p = this->sGenNodesCoeff + chunkIx*this->sizeGenNodeCoeff;

    for (int i = 0; i < nAlloc; i++) {
        newNode_cp->serialIx = *serialIx+i;//Until overwritten!
        if (this->genNodeStackStatus[*serialIx+i] != 0)
	    println(0, *serialIx+i<<" NodeStackStatus: not available " << this->genNodeStackStatus[*serialIx+i]);
        this->genNodeStackStatus[*serialIx+i] = 1;
        newNode_cp++;
    }
    this->nGenNodes += nAlloc;
    this->lastGenNode += nAlloc;
    
    omp_unset_lock(&Stree_lock);
    return newNode;
}

template<int D>
void SerialTree<D>::deallocGenNodes(int serialIx) {
    omp_set_lock(&Stree_lock);
    if (this->nGenNodes <0) {
        println(0, "minNodes exceeded " << this->nGenNodes);
        this->nGenNodes++;
    }
    this->genNodeStackStatus[serialIx] = 0;//mark as available
    if (serialIx == this->nGenNodes-1) {//top of stack
        int topStack = this->nGenNodes;
        while (this->genNodeStackStatus[topStack-1] == 0) {
            topStack--;
            if (topStack < 1) break;
        }
        this->nGenNodes = topStack;//move top of stack
        //has to redefine lastGenNode
        int chunk = this->nGenNodes/this->maxNodesPerChunk;//find the right chunk
        this->lastGenNode = this->genNodeChunks[chunk] + this->nGenNodes%(this->maxNodesPerChunk);
    }
    omp_unset_lock(&Stree_lock);
 }

/** SerialTree destructor. */
template<int D>
SerialTree<D>::~SerialTree() {
    for (int i = 0; i < this->genNodeCoeffChunks.size(); i++) delete[] this->genNodeCoeffChunks[i];
    for (int i = 0; i < this->nodeChunks.size(); i++) delete[] (char*)(this->nodeChunks[i]);
    for (int i = 0; i < this->nodeCoeffChunks.size(); i++) delete[] this->nodeCoeffChunks[i];
    for (int i = 0; i < this->genNodeChunks.size(); i++) delete[] (char*)(this->genNodeChunks[i]);

    delete[] this->nodeStackStatus;
    delete[] this->genNodeStackStatus;
}

template class SerialTree<1>;
template class SerialTree<2>;
template class SerialTree<3>;
