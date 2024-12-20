#pragma once
#include "evaluate.hpp"
#include "cache.hpp"

enum moveType{
    hashMove = 4,
    justEatMove = 3,
    killerMove = 2,
    historyMove = 1
};

enum rootMoveType{
    nonFindPv = 1,
    findPv = 2
};

class moveSort{
public:
    static void sortNormalMoveSeuqance(evaluate& e,historyCache& h,vector<step>& moveList){
        static int i,a = 0;
        for(step& move : moveList){
            const int vlEat = getMvvLva(e,move);
            if(vlEat >= 0){
                move.sortType = justEatMove;
                move.vl = vlEat;
            }else{
                move.sortType = historyMove;
                move.vl = h.getCache(move);
            }
        }
        sort(moveList.begin(),moveList.end(), vlCompare);
    }
    static void sortQuesicMoveSequance(evaluate& e,vector<step>& moveList){
        for(step& move : moveList){
            if(move.toPiece){
                move.vl = getMvvLva(e,move);
            }
        }
        sort(moveList.begin(),moveList.end(), vlCompare);
    }
    static void initSortRootMoveSeuqance(evaluate& e,vector<step>& moveList){
        vector<step> tempMoveList;
        for(step& move : moveList){
            if(e.makeMove(move.fromPos,move.toPos)){
                tempMoveList.push_back(move);
                e.unMakeMove();
            }
        }
        swap(tempMoveList,moveList);
        for(step& move : moveList){
            move.sortType = nonFindPv;
            int vlEat = getMvvLva(e,move);
            if(vlEat >= 0){
                move.vl = vlEat;
            }
        }
        sort(moveList.begin(),moveList.end(), vlCompare);
    }
    static void refreshRootMoveSequance(vector<step>& moveList,historyCache& h,step& betterMove){
        for(step& move : moveList){
            if(move == betterMove){
                move.sortType = findPv;
                move.vl = SORT_MAX_VALUE;
            }else if(move.sortType == findPv){
                move.vl--;
            }
        }
    }
    static void sortRootMoveSequance(vector<step>& moveList){
        sort(moveList.begin(),moveList.end(), vlAndTypeCompare);
    }
private:
    static int inOtherStepList(step& move,vector<step>& moveList){
        for(int i = 0;i < moveList.size();i++){
            if(move == moveList[i]){
                return i + 1;
            }
        }
        return 0;
    }
    static int getMvvLva(evaluate& e,step& move){
        assert(move.toPiece);
        if(genMove::getRelation(e,move.toPos,move.toPiece,beProtected)){
            const int lva = vlMvvLva[abs(move.fromPiece) - 1];
            const int mvv = vlMvvLva[abs(move.toPiece) - 1];
            const int toType = swapBasicBoard::pieceToAbsType(move.toPiece);
            if(mvv > lva){
                return mvv - lva + 1;
            }
            if(mvv == lva && lva >= 4){
                return 1;
            }
            if(inSideBoard[move.toPos] * move.toPiece < 0 && toType == pawn){
                return 1;
            }
            return mvv - lva;
        }
        return vlMvvLva[abs(move.toPiece) - 1];

    }
    static bool vlAndTypeCompare(const step& firstMove,const step& secondMove){
        if(firstMove.sortType != secondMove.sortType) {
            return firstMove.sortType > secondMove.sortType;
        }
        return firstMove.vl > secondMove.vl;
    }
    static bool vlCompare(const step& firstMove,const step& secondMove){
        return firstMove.vl > secondMove.vl;
    }
    friend class searchGroup;
};

class searchGroup{
public:
    searchGroup(){
        pvLine = vector<int>(0,MAX_MOVE_NUM);
    }
public:
    int searchQuesic(evaluate& e,int vlAlpha,int vlBeta) {
        int vl = harmlessPruning(e,vlBeta);
        if(vl > MIN_VALUE){
            return vl;
        }

        if(e.getNowDistance() >= MAX_QUIESC_DISTANCE){
            return e.getEvaluate(e.side,vlAlpha,vlBeta);
        }

        int vlBest = MIN_VALUE;
        vector<step> moveList;
        const bool bCheck = e.checkMoveStatus.back();
        if(bCheck){
            genMove::genMoveList(e,moveList,all);
        }else{
            vl = e.getEvaluate(e.side,vlAlpha,vlBeta);
            if(vl >= vlBeta){
                return vl;
            }
            vlBest = vl;
            vlAlpha = max(vl,vlAlpha);

            genMove::genMoveList(e, moveList,justEat);
            moveSort::sortQuesicMoveSequance(e,moveList);
        }
        for(step& move : moveList){
            if(!bCheck && move.vl <= 0){
                break;
            }
            if(e.makeMove(move.fromPos,move.toPos)){
                vl = -searchQuesic(e,-vlBeta,-vlAlpha);
                e.unMakeMove();

                if(vl > vlBest){
                    if(vl >= vlBeta){
                        return vl;
                    }
                    vlBest = vl;
                    vlAlpha = max(vl,vlAlpha);
                }
            }
        }
        if(vlBest == MIN_VALUE){
            return MIN_VALUE + e.getNowDistance();
        }
        return vlBest;
    }
    int searchPV(evaluate& e,int depth,int vlAlpha,int vlBeta,bool noNull = false){
        if(depth <= 0){
            return searchQuesic(e,vlAlpha,vlBeta);
        }

        int vl = harmlessPruning(e,vlBeta);
        if(vl > MIN_VALUE){
            return vl;
        }
        int vlBest = MIN_VALUE;
        int nodeType = alpha;
        vector<step> moveList;
        step* pBestMove = nullptr;
        const bool bCheck = !e.checkMoveStatus.empty() && e.checkMoveStatus.back();
        int newDepth = bCheck ? depth : depth - 1;

        bool quit = false;

        tinyMove tMove;
        step convert_move;

        if(hashMap.getCache(e,depth,vlAlpha,vlBeta,vl,tMove)){
            if(e.stable()){
                return vl;
            }else{
                if(vl >= vlBeta && searchQuesic(e,vlBeta - 1,vlBeta) >= vlBeta){
                    return vl;
                }else if(vl <= vlAlpha && searchQuesic(e,vlAlpha,vlAlpha + 1) <= vlAlpha){
                    return vl;
                }else if(vl > vlAlpha && vl < vlBeta){
                    if(searchQuesic(e,vlAlpha,vlAlpha + 1) > vlAlpha && searchQuesic(e,vlBeta - 1,vlBeta) < vlBeta){
                        return vl;
                    }
                }else if(bCheck){
                    newDepth --;
                }
            }
        }

        if(!tMove.fromPos && newDepth >= 2){
            vl = searchPV(e,newDepth / 2,vlAlpha,vlBeta);
            if(vl <= vlAlpha){
                vl = searchPV(e,newDepth / 2,MIN_VALUE,vlBeta);
            }
            hashMap.getCacheMove(e,tMove);
        }


        //置换表启发
        if (genMove::legalMove(e, convert_move)) {
            if (e.makeMove(convert_move.fromPos, convert_move.toPos)) {
                vl = -searchPV(e, newDepth, -vlBeta, -vlAlpha);
                e.unMakeMove();

                if(vl > vlBest){
                    vlBest = vl;
                    if(vl >= vlBeta){
                        pBestMove = &convert_move;
                        nodeType = beta;
                        quit = true;
                    }
                    if(vl > vlAlpha){
                        pBestMove = &convert_move;
                        nodeType = pv;
                        vlAlpha = vl;
                    }
                }
            }
        }

        //截断启发
        vector<step> killerMoveList;
        if(!quit){
            killerMap.getCache(e,killerMoveList);
            for(step & move : killerMoveList){
                if(move != convert_move){
                    if(e.makeMove(move.fromPos,move.toPos)){
                        if(vlBest == MIN_VALUE){
                            vl = -searchPV(e, newDepth,-vlBeta,-vlAlpha);
                        }else{
                            vl = -searchNonPV(e,newDepth,-vlAlpha);
                            if(vl > vlAlpha && vl < vlBeta){
                                vl = -searchPV(e,newDepth,-vlBeta,-vlAlpha);
                            }
                        }
                        e.unMakeMove();

                        if(vl > vlBest){
                            vlBest = vl;
                            if(vl >= vlBeta){
                                quit = true;
                                pBestMove = &move;
                                nodeType = beta;
                                break;
                            }
                            if(vl > vlAlpha){
                                pBestMove = &move;
                                vlAlpha = vl;
                                nodeType = pv;
                            }
                        }
                    }
                }
            }
        }

        //剩余走法
        if(!quit){
            genMove::genMoveList(e,moveList,all);
            moveSort::sortNormalMoveSeuqance(e,historyMap,moveList);
            for(step & move : moveList){
                if(!moveSort::inOtherStepList(move,killerMoveList) &&
                    move != convert_move){
                    if(e.makeMove(move.fromPos,move.toPos)){
                        if(vlBest == MIN_VALUE){
                            vl = -searchPV(e, newDepth,-vlBeta,-vlAlpha);
                        }else{
                            vl = -searchNonPV(e,newDepth,-vlAlpha);
                            if(vl > vlAlpha && vl < vlBeta){
                                vl = -searchPV(e,newDepth,-vlBeta,-vlAlpha);
                            }
                        }
                        e.unMakeMove();

                        if(vl > vlBest){
                            vlBest = vl;
                            if(vl >= vlBeta){
                                pBestMove = &move;
                                nodeType = beta;
                                quit = true;
                                break;
                            }
                            if(vl > vlAlpha){
                                pBestMove = &move;
                                nodeType = pv;
                                vlAlpha = vl;
                            }
                        }
                    }
                }
            }
        }

        if(pBestMove){
            hashMap.recoardCache(e,nodeType,vlBest,depth,pBestMove);
            historyMap.recoardCache(*pBestMove,depth);
            if(nodeType != beta){
                killerMap.recoardCache(e,*pBestMove);
            }
        }else{
            hashMap.recoardCache(e,alpha,vlBest,depth, nullptr);
        }

        if(vlBest == MIN_VALUE){
            return MIN_VALUE + e.getNowDistance();
        }
        return vlBest;
    }
    int searchNonPV(evaluate& e,int depth,int vlBeta,bool noNull = false){
        if(depth <= 0){
            return searchQuesic(e,vlBeta - 1,vlBeta);
        }

        int vl = harmlessPruning(e,vlBeta);
        if(vl > MIN_VALUE){
            return vl;
        }
        int vlBest = MIN_VALUE;
        step* pBestMove = nullptr;
        const bool bCheck = !e.checkMoveStatus.empty() && e.checkMoveStatus.back();
        int newDepth = bCheck ? depth : depth - 1;

        tinyMove tMove;
        if(hashMap.getCache(e,depth,vlBeta - 1,vlBeta,vl,tMove)){
            if(e.stable()){
                return vl;
            }else{
                const int svl = searchQuesic(e,vlBeta - 1,vlBeta);
                if(vl >= vlBeta && svl >= vlBeta){
                    return vl;
                }else if(vl < vlBeta && svl < vlBeta){
                    return vl;
                }else if(bCheck){
                    newDepth --;
                }
            }
        }

        //空着启发
        if(!noNull && !bCheck && e.nullOkay()){
            if(e.makeNullMove()){
                vl = -searchNonPV(e,depth - NULL_DEPTH - 1,-vlBeta + 1,true);
                e.unMakeNullMove();

                if(vl >= vlBeta){
                    if(e.nullSafe()){
                        hashMap.recoardCache(e,beta,vl,max(depth,NULL_DEPTH + 1));
                        return vl;
                    }else if(searchNonPV(e,depth - NULL_DEPTH,vlBeta,true) >= vlBeta){
                        hashMap.recoardCache(e,beta,vl,max(depth,NULL_DEPTH));
                        return vl;
                    }
                }
            }
        }

        //置换表启发
        bool quit = false;
        step convert_move = step(tMove.fromPos,tMove.toPos,tMove.fromPiece,tMove.toPiece);
        if (genMove::legalMove(e, convert_move)) {
            if (e.makeMove(convert_move.fromPos, convert_move.toPos)) {
                vl = -searchNonPV(e, newDepth, -vlBeta + 1);
                e.unMakeMove();

                if (vl > vlBest) {
                    vlBest = vl;
                    if (vl >= vlBeta) {
                        pBestMove = &convert_move;
                        quit = true;
                    }
                }
            }
        }

        //截断启发
        vector<step> killerMoveList;
        if(!quit){
            killerMap.getCache(e,killerMoveList);
            for(step & move : killerMoveList){
                if(move != convert_move){
                    if(e.makeMove(move.fromPos,move.toPos)){
                        vl = -searchNonPV(e,newDepth,-vlBeta + 1);
                        e.unMakeMove();

                        if(vl > vlBest){
                            vlBest = vl;
                            if(vl >= vlBeta){
                                pBestMove = &move;
                                quit = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        //吃子启发
        vector<step> moveList;
        if(!quit){
            genMove::genMoveList(e,moveList,all);
            moveSort::sortNormalMoveSeuqance(e,historyMap,moveList);
            for(step & move : moveList){
                if(!moveSort::inOtherStepList(move,killerMoveList) &&
                    move != convert_move){
                    if(e.makeMove(move.fromPos,move.toPos)){
                        vl = -searchNonPV(e,newDepth,-vlBeta + 1);
                        e.unMakeMove();

                        if(vl > vlBest){
                            vlBest = vl;
                            if(vl >= vlBeta){
                                pBestMove = &move;
                                quit = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if(pBestMove){
            hashMap.recoardCache(e,beta,vlBest,depth,pBestMove);
            historyMap.recoardCache(*pBestMove,depth);
            killerMap.recoardCache(e,*pBestMove);
        }else{
            hashMap.recoardCache(e,alpha,vlBest,depth, nullptr);
        }

        if(vlBest == MIN_VALUE){
            return MIN_VALUE + e.getNowDistance();
        }
        return vlBest;
    }
    int searchRoot(evaluate& e,int maxDepth){
        int vl;
        int vlBest = MIN_VALUE;
        const bool bCheck = !e.checkMoveStatus.empty() && e.checkMoveStatus.back();
        moveSort::sortRootMoveSequance(rootMoveList);
        for(step & move : rootMoveList){
            const int newDepth = bCheck ? maxDepth : maxDepth - 1;
            if(e.makeMove(move.fromPos,move.toPos)){
                if(vlBest == MIN_VALUE){
                    vl = -searchPV(e, newDepth,MIN_VALUE,MAX_VALUE);
                }else{
                    vl = -searchNonPV(e,newDepth,-vlBest);
                    if(vl > vlBest){
                        vl = -searchPV(e,newDepth,MIN_VALUE,-vlBest);
                    }
                }
                e.unMakeMove();

                if(vl > vlBest){
                    move.vl = vl;
                    moveSort::refreshRootMoveSequance(rootMoveList,historyMap,move);
                    vlBest = vl;
                }
            }
        }

        if(!rootMoveList.empty()){
            step& bestMove = rootMoveList.front();
            hashMap.recoardCache(e,pv,vlBest,maxDepth,&bestMove);
            historyMap.recoardCache(bestMove,maxDepth);
            bestMove.printMove();
        }
        return vlBest;
    }
    int searchMain(evaluate& e,int maxDepth,int maxTime){
        //init
        initSearch(e);
        //init para
        int vlBest = MIN_VALUE;
        //search
        clock_t start = clock();
        for(int depth = 1;depth <= maxDepth;depth++){
            int vl = searchRoot(e,depth);
            if(vl > vlBest){
                vlBest = vl;
                //toDo something
            }
            clock_t now = clock();
            cout<<"depth = "<<depth<<" | vl = "<<vl<<" | time_sum =  "<<setprecision(3)<<(double)(now - start) / CLOCKS_PER_SEC<<"s"<<endl;
            if(now - start >= maxTime / 3){
                break;
            }
        }
        return vlBest;
    }
protected:
    void initSearch(evaluate& e){
        e.resetEvaBoard();
        historyMap.clearCache();
        killerMap.clearCache();
        hashMap.clearCache();
        vector<step>().swap(rootMoveList);
        genMove::genMoveList(e,rootMoveList,all);
        moveSort::initSortRootMoveSeuqance(e,rootMoveList);
    }
private:
    static int harmlessPruning(evaluate& e,int vlBeta){
        //杀棋步数裁剪
        const int nDistance = e.getNowDistance();
        int vl = MIN_VALUE + nDistance;
        if(vl >= vlBeta){
            return vl;
        }

        //自然限着裁剪
        if(e.isDraw()){
            return 0;
        }

        //重复走法路线裁剪
        const int repType = e.isRep();
        if(repType == draw_rep){
            return e.getDrawValue();
        }else if(repType == kill_rep){
            return MAX_BAN_VALUE - nDistance;
        }else if(repType == killed_rep){
            return MIN_BAN_VALUE + nDistance;
        }
        return MIN_VALUE;
    }
private:
    historyCache historyMap;        //历史启发表
    killerCache killerMap;          //截断启发表
    hashCache hashMap;              //哈希启发表
    vector<int> pvLine;             //主要遍历路线
    vector<step> rootMoveList;      //根节点走法表
    friend class pipe;
};