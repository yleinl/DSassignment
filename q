[33mcommit b6f9ff23c17f838c30d216c61397296c509f4b8a[m[33m ([m[1;36mHEAD -> [m[1;32musers/zhiyan/distributed_GCN[m[33m, [m[1;31morigin/master[m[33m, [m[1;31morigin/HEAD[m[33m, [m[1;32mmaster[m[33m)[m
Author: abnerlyt <644044443@qq.com>
Date:   Tue Dec 5 05:58:05 2023 +0100

    异步通信utils，先send好size再irecv

[33mcommit 26b6690f2587768d15bbfc8b29870ef0dcc8754e[m
Author: abnerlyt <644044443@qq.com>
Date:   Tue Dec 5 04:30:43 2023 +0100

    标记了需要通信的节点: 需要发送的目标以及需要接受的来源

[33mcommit 8ac13b36d8016b92da130d617630bb8ef9124b40[m
Author: abnerlyt <644044443@qq.com>
Date:   Tue Dec 5 00:04:42 2023 +0100

    保留原来的edge_index和补充所有节点所属的分区

[33mcommit 4ca92e15cde3851e037b5f65ff9b73bef232b2ce[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Dec 4 16:23:22 2023 +0100

    改错

[33mcommit 66ccb1f0b77dcca0344e4fa5e9f0ef6b4d7822a3[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Dec 4 15:56:09 2023 +0100

    改错

[33mcommit fee486deb7b05d5142b43bad521cc46e785c4dea[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Dec 4 15:46:06 2023 +0100

    忘记import了

[33mcommit 48ab166c7783daf1e057ab4e719f57f94175cabe[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Dec 4 15:38:00 2023 +0100

    去掉冗余点

[33mcommit f07fbf692472518775ec1c8cebaf1811a4d1bf64[m
Merge: 157fad8 f4db744
Author: abnerlyt <644044443@qq.com>
Date:   Sun Dec 3 21:00:17 2023 +0100

    Merge remote-tracking branch 'origin/users/zhiyan/distributed_GCN'

[33mcommit f4db744c11c0e9106dd4255f798f7b9955ffd708[m[33m ([m[1;31morigin/users/zhiyan/distributed_GCN[m[33m)[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Sun Dec 3 20:32:20 2023 +0100

    done missing node feature communication between 2 nodes

[33mcommit b4ae0e4fcbdd64078b871d3321d265f589f2d6b1[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Sun Dec 3 20:30:55 2023 +0100

    number of nodes changed back to 3

[33mcommit 157fad85c4f564bd038578bc0f0593141e19ced3[m
Author: abnerlyt <644044443@qq.com>
Date:   Sun Dec 3 18:35:53 2023 +0100

    剩下两个数据集也训练好了，删了没用的东西

[33mcommit 06467eb0b102b9e2379baf01fe8739b4634098ab[m
Author: abnerlyt <644044443@qq.com>
Date:   Sat Dec 2 20:18:11 2023 +0100

    problem fix

[33mcommit 2daa7f7dd9f744d5a4e15af2ddf53027530b08e7[m
Author: abnerlyt <644044443@qq.com>
Date:   Sat Dec 2 20:03:40 2023 +0100

    Add sampler for Reddit & Yelp training

[33mcommit c2aff65a7a11ccba4cb886968da3ebedf7caac0e[m
Author: abnerlyt <644044443@qq.com>
Date:   Sat Dec 2 17:33:37 2023 +0100

    实现了partition_data_louvain方法，减少通信有负载均衡，需要安装louvain相关依赖

[33mcommit 11f29296a352e17ceead1978acfdccf92ca8867f[m
Author: abnerlyt <644044443@qq.com>
Date:   Sat Dec 2 17:28:04 2023 +0100

    实现了partition_data_louvain方法，减少通信有负载均衡，需要安装louvain相关依赖

[33mcommit dc303f933a8b3d866ee74f3887ad59b12028bf20[m
Merge: dfdd976 00dc8f9
Author: abnerlyt <644044443@qq.com>
Date:   Fri Dec 1 20:56:47 2023 +0100

    Merge remote-tracking branch 'origin/master'

[33mcommit dfdd976591d1c1666efd7d975da28eee8f50a95b[m
Author: abnerlyt <644044443@qq.com>
Date:   Fri Dec 1 20:54:55 2023 +0100

    heartbeat 故障监听

[33mcommit 00dc8f9f794666d38f6672158d6690d8e9764780[m
Merge: c0d57b5 51f4d6d
Author: Zhiyan Yao <rachel2804195717@icloud.com>
Date:   Fri Dec 1 16:13:23 2023 +0100

    Merge pull request #2 from yleinl/users/zhiyan/distributed_GCN
    
    Users/zhiyan/distributed gcn

[33mcommit 51f4d6dec475dccbd886d784ad957ea23a4c88a6[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Fri Dec 1 16:08:43 2023 +0100

    enabled inference on node 0

[33mcommit e22f8c00ea055db22881a868eed7065e270785f3[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Fri Dec 1 16:07:31 2023 +0100

    number of nodes changed from 3 to 2, which is the simplest case

[33mcommit ad3604edf0d813c971137360187e4a76be3aad6c[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Fri Dec 1 16:06:32 2023 +0100

    output file deleted

[33mcommit c0d57b51bd394e2c213434e1226eafc14fe2c9d1[m
Author: abnerlyt <644044443@qq.com>
Date:   Wed Nov 29 23:14:09 2023 +0100

    删了点没用的，改了个小错

[33mcommit 8a6bbd92a9a3afac95cb25f17d836f83b9299bfc[m
Author: abnerlyt <644044443@qq.com>
Date:   Wed Nov 29 22:59:43 2023 +0100

    _Distributed.py写错了，这个是本机跑的，DAS上跑MulNode那个

[33mcommit d8ad7b63ccbba78b879e5e6a1c64ca122e6f2dfc[m
Author: abnerlyt <644044443@qq.com>
Date:   Wed Nov 29 22:52:40 2023 +0100

    合并到_MulNode.py里面了，_Distributed是单机多线程模拟用的，保留一个GCN.py原始模型

[33mcommit 5b7c0e4d99b571f573ebdbfe0bbf9ba213c0eefc[m
Merge: b709556 017904b
Author: Zhiyan Yao <rachel2804195717@icloud.com>
Date:   Wed Nov 29 21:17:30 2023 +0100

    Merge pull request #1 from yleinl/users/zhiyan/distributed_GCN
    
    Users/zhiyan/distributed gcn

[33mcommit 017904b10830972db9a308323d6a80d6b9579a1b[m
Merge: a71faf3 b709556
Author: Zhiyan Yao <rachel2804195717@icloud.com>
Date:   Wed Nov 29 21:17:15 2023 +0100

    Merge branch 'master' into users/zhiyan/distributed_GCN

[33mcommit a71faf3f2a7de60accb62ddf58fd6a1f7e02b07c[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Wed Nov 29 21:02:59 2023 +0100

    communication on 2 nodes

[33mcommit b709556bfe5dc7c10fdaf9138c676809934ef7f4[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Tue Nov 28 21:32:08 2023 +0100

    implement the distributed on multiple nodes

[33mcommit 9956bb2f2a7bd88965a3f64be6a27fd263d04192[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Nov 27 23:00:26 2023 +0100

    加入了多节点和slurm文件，更新requirement

[33mcommit ee22defb7a60936c4221aa58049d376a340796cd[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Nov 27 22:17:26 2023 +0100

    删了点没用的东西

[33mcommit 15612c01fe98408107caebd5ec92d95fe7ac356c[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Nov 27 22:17:04 2023 +0100

    删了点没用的东西

[33mcommit 4b8fc549b52186b9ee6348811b81cdef43d96e1f[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Nov 27 20:41:56 2023 +0100

    包括冗余节点的数据划分

[33mcommit 5f214a4f68f05131207795f79af8376631fe104b[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Nov 27 20:40:23 2023 +0100

    没包括冗余节点的数据划分

[33mcommit 551fc7c775b129d00f6f78f438601ea5366245b2[m
Author: DSYS course <dsys2352@fs0.das6.cs.vu.nl>
Date:   Mon Nov 27 19:23:09 2023 +0100

    requirements.txt updated

[33mcommit f8e0aea5d782c9fe61f5bb8f86ab98bce63d9c48[m
Author: abnerlyt <644044443@qq.com>
Date:   Mon Nov 27 18:13:42 2023 +0100

    最简单的数据划分

[33mcommit d1bb6da8da95fe0109c3f25b6c4259c9fe621d98[m[33m ([m[1;31morigin/DistributedWorkspace[m[33m)[m
Author: 雷义焘 <8231658+lei-yitao@user.noreply.gitee.com>
Date:   Sat Nov 18 17:25:27 2023 +0100

    包含了pubmed和cora的参数，不过自己训练一下也一样的，测试的时候把optimizer后面的注掉，解注model.load_state_dict就行

[33mcommit 5aa866dbe97a57e87b15ebaaa43fd597e5a88cc6[m
Author: yleinl <yitao.lei@student.uva.nl>
Date:   Sat Nov 18 14:15:34 2023 +0100

    改了下epoch数量

[33mcommit 3a97d4d43137d7b86a16b9c657743428c11a7b16[m
Author: yleinl <yitao.lei@student.uva.nl>
Date:   Sat Nov 18 14:15:30 2023 +0100

    改了下epoch数量

[33mcommit 2044aff59e4c494ec941c6d20b88496dfd415059[m
Author: yleinl <yitao.lei@student.uva.nl>
Date:   Sat Nov 18 14:03:52 2023 +0100

    添加Yelp数据集

[33mcommit 8df43e15fa798e734885207c3219f30cfae12a29[m
Author: yleinl <yitao.lei@student.uva.nl>
Date:   Sat Nov 18 12:33:42 2023 +0100

    改成我们需要用的版本

[33mcommit e3ede10d9de0b3849d0a5180a92233c0b42375a9[m
Author: Alfred Feldmeyer <6654064+byteSamurai@users.noreply.github.com>
Date:   Sun May 30 18:55:34 2021 +0200

    Fix diagonal degree matrix creation (#14)

[33mcommit f5a10700065769737a5630025d3d6fe0cc37711c[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Wed May 19 15:58:59 2021 +0530

    Add cite entry for blog

[33mcommit 3446d9f53ad63e4871c621b2c1902595ad1ecebe[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Thu Sep 17 06:54:28 2020 +0530

    Fix broken chebnet link

[33mcommit 4043d2e929e872dd1c235343fd7a33f568dc705f[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Sat May 23 11:11:48 2020 +0530

    Update GraphSAGE.py

[33mcommit dbbcaff1573ccfb255c033eac8a2e7107dbdc203[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Mon Feb 24 20:11:16 2020 +0530

    add cover img

[33mcommit b72953a412053759fd9c5e50bc2a23834216a97a[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Feb 9 19:47:33 2020 +0530

    add assets

[33mcommit 98239be4f8a76234a9d4bf1282d61168556a909a[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Feb 9 19:44:07 2020 +0530

    add requirements.txt

[33mcommit e462b11f41a1bb607c12f3d588ea377e2a08f2f3[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Wed Feb 5 07:24:29 2020 +0530

    fix rendering issues

[33mcommit c428dc8600c5eb00e0ef76aefc7b4a446f3d1230[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Wed Feb 5 07:16:03 2020 +0530

    fix line breaks

[33mcommit e75ed1488ee22132480ae1d87a1fbde4b9cd0a9c[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Wed Feb 5 07:00:00 2020 +0530

    Update README.md

[33mcommit be037ef5d9a69339bd243fa1556eac57dcc664e1[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 07:43:30 2020 +0530

    update graphsage blog

[33mcommit f2c1b584ae2badaf97d6ccb0b505eeb3ea50274e[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 07:43:02 2020 +0530

    update chebnet

[33mcommit 88d776f286f08f60a19cedd604b0cf9d7b7e13c7[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 01:27:56 2020 +0530

    fix deepwalk

[33mcommit 016320df4ec84345e4ab095fcb4ec44bf54c6782[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 00:37:57 2020 +0530

    remove redundant

[33mcommit 55dac170aa09ce90eb5e3e033ba0c496c0e58ccb[m
Merge: 6733cbd 6f1a7e4
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 00:22:11 2020 +0530

    Merge branch 'master' of https://github.com/dsgiitr/graph_nets

[33mcommit 6733cbd5c2534da09fa172efdda9dfe811088c3f[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 00:21:39 2020 +0530

    fix matplotlib animation

[33mcommit 01c103b776eab7ee887fcbceb6ccbc8c39503cc0[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 00:21:00 2020 +0530

    add train karate & restructure

[33mcommit a04da81dd5b2f251c11b7b0cc17f67da759cec9f[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Feb 4 00:16:15 2020 +0530

    add GCN code

[33mcommit 6f1a7e4cd61978f78e3d6422d0c4dc45b0b482ee[m
Merge: a0af829 bc08ae5
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Sat Feb 1 14:20:40 2020 +0530

    Merge pull request #10 from dsgiitr/graphsage
    
    Graphsage

[33mcommit bc08ae51452bf8ea674e83c045716ce963bf4b53[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Feb 1 14:19:44 2020 +0530

    restructure and fix redundant

[33mcommit a0af8294e58e9e6494f88af2bb0cf4ed2dbb8478[m
Merge: 2fae4b8 353a7c2
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Sat Feb 1 05:11:57 2020 +0530

    Merge pull request #9 from dsgiitr/chebnet
    
    merge branch chebnet

[33mcommit 2fae4b8dc885ea07ed134078fa1005a3c2a864c0[m
Merge: 8d1744b 64b5396
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Sat Feb 1 05:10:45 2020 +0530

    Merge pull request #8 from dsgiitr/deepWalk
    
    Deep walk

[33mcommit 8d1744b10724d52ac1865583e821d2d498cfefcf[m
Merge: 30c0453 d802b3c
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Sat Feb 1 05:10:12 2020 +0530

    Merge pull request #7 from dsgiitr/GCNConv_blog
    
    Gcn conv blog

[33mcommit 30c0453866d7311e6f8aa144c620039c93b36aa6[m
Merge: 91b58dd 853dfbc
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Sat Feb 1 05:07:52 2020 +0530

    Merge pull request #6 from dsgiitr/GAT
    
    Gat

[33mcommit d802b3c90be70ceafc64345b6966cc8c099292e8[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Feb 1 03:29:14 2020 +0530

    Link to Geometric implementation is missing,
    
    Added a reminder for that

[33mcommit ff8c275c9992cbefa999f7768c052afb8ab8ba50[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Feb 1 03:24:20 2020 +0530

    Added images

[33mcommit f15426b3fa967469c6b75716acc226a277c0741c[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Feb 1 03:23:37 2020 +0530

    Adding images

[33mcommit 6626d2faca8c621deecdb50038d48671a35b6504[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Feb 1 03:22:32 2020 +0530

    Updating images

[33mcommit 06a56b3c2287e00ccb9d1ffb58ef06026ff2033c[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Feb 1 03:21:39 2020 +0530

    Create README.md

[33mcommit c00be689224c171021969b4624cb675ddef099d4[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Feb 1 03:19:24 2020 +0530

    updated GCN.md

[33mcommit f1e3c948e3cf933779c16c30353b87999468c36e[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Feb 1 03:00:13 2020 +0530

    updated blog

[33mcommit 64b5396b9350b63e26e69421c8a1fb4d47ca15b7[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Fri Jan 31 23:35:52 2020 +0530

    Update DeepWalk.md

[33mcommit 460091010298eb5ba9a5e6a2e232a51bfba6d530[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Fri Jan 31 23:32:57 2020 +0530

    Adding reviewed md file of blog

[33mcommit 853dfbce7c791b134ca47d4d5e2807708068cd7c[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Fri Jan 31 21:45:18 2020 +0530

    GAT Reviewed

[33mcommit 8353d3db799ce43e57906f9ec8dfdb373f89195c[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Fri Jan 31 21:42:17 2020 +0530

    GAT Reviewed

[33mcommit 4eec8bdc5bb9106781f2aa911d182bff1bc11cf7[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Fri Jan 31 20:52:23 2020 +0530

    Updated Blog

[33mcommit a1c96fd406b453e8770845b064be75b90e2e97d9[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 28 20:35:56 2020 +0530

    Add files via upload

[33mcommit 564132097bc1e318f1201a0a69c0f65acb645dda[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 28 20:35:01 2020 +0530

    Create README

[33mcommit 353a7c2121b4b86e0c4525d5693223f62e7e5a14[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Jan 28 20:34:27 2020 +0530

    restructuring

[33mcommit 94ebbaf53a985776d44350c6066c36bf562eb886[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Jan 28 20:30:24 2020 +0530

    fix markdown rendering issues

[33mcommit 344f254f6a7003cc1f1eba58655f5f171a0c37d7[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 28 20:28:41 2020 +0530

    Minor changes
    
    Add GCN markdown

[33mcommit 6e4d1540254e948ae0629200c30fc1038f714299[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 28 20:09:19 2020 +0530

    Rename Blog(Incomplete).ipynb to Blog.ipynb

[33mcommit 5a3d94dfd31e4c1de4109ea032bc2d2e0bcf60f9[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 28 20:07:27 2020 +0530

    updated cora

[33mcommit b446025d254b7b5b70448f659f47f5f7f44667c7[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 28 20:06:58 2020 +0530

    Create README

[33mcommit b80b712a7a95c666e526814c04a8cdcb402f0b32[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 28 20:05:36 2020 +0530

    Added code

[33mcommit 91b58dd74bc13b7b953714bcf590e4dbb6a99768[m
Merge: 376e762 0e7765a
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Tue Jan 28 18:40:18 2020 +0530

    Merge pull request #4 from dsgiitr/revert-3-chebnet
    
    Revert "Implementation of Chebnet"

[33mcommit 0e7765a65fbdae4b062fe22f94857e39ecfa1fa7[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Tue Jan 28 18:40:05 2020 +0530

    Revert "Implementation of Chebnet"

[33mcommit 376e762497890722012e485633dccf3f8ab554bc[m
Merge: 92e3049 3f266c0
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Tue Jan 28 18:38:56 2020 +0530

    Merge pull request #3 from dsgiitr/chebnet
    
    Implementation of Chebnet

[33mcommit 3f266c042741d10bec0acaa6089aa3c03cd2cf61[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Tue Jan 28 18:38:10 2020 +0530

    Implementation of Chebnet

[33mcommit fb7d1bbc66d2933e5d110af15d5440f12c5ad31e[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Mon Jan 27 16:35:20 2020 +0530

    add markdown

[33mcommit afe10ae2eba00c5564cc07399df737b8e36dbe04[m
Author: shubhamgit1 <shubham22019@gmail.com>
Date:   Tue Jan 14 23:06:28 2020 +0530

    made some corrections

[33mcommit ec340509dea69ed9c6e3289fce31923c681f2ce4[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 14 15:21:55 2020 +0530

    Added Loss image

[33mcommit 708ac304662930b582fcaab3a500d85a4648fe79[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Jan 14 15:21:03 2020 +0530

    Combined All

[33mcommit 3218abf611bc457eb02a7bcf0a7abeb384e6b947[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Tue Dec 31 08:05:31 2019 +0530

    minor update

[33mcommit 387c1ece2d0aae9a9281ba48b21e2bf84b8e608e[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Dec 28 13:58:11 2019 +0530

    add assets

[33mcommit 81ed01428fc7be1852fae779dd1477151c89908c[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Dec 28 13:57:36 2019 +0530

    add initial blog

[33mcommit 92b1f2c2a203cf31012a2f2f4afcf46bb1ebc445[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Dec 28 07:52:09 2019 +0530

    minor change

[33mcommit 808d7d907bfd0f6475ce6cdc67c48057901d32a9[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Dec 28 03:59:03 2019 +0530

    add cnn vs gcn

[33mcommit 02e574e3a940f9032768c07850d9f766a85010bf[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Dec 28 03:40:34 2019 +0530

    add pyg code gat

[33mcommit a431bd92444a3e65e89bf801e1e600a89ecd7005[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Dec 15 11:00:05 2019 +0530

    fix math eq

[33mcommit 4f28ae62ad343da8b0868665a13181853d815243[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Dec 15 10:04:30 2019 +0530

    fix dir structure

[33mcommit 3cab7419fe2439547d37f3e689d3be00258fe38e[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Dec 15 10:02:43 2019 +0530

    add architecture asset

[33mcommit 5b68016533695ed8466dcbe6cfae8936f256dda6[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Dec 15 09:59:29 2019 +0530

    add assets

[33mcommit d262a5e533423f7f0828af5ea294b7d9e5dea3e2[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Dec 15 09:58:37 2019 +0530

    update blog+code

[33mcommit d7dfe439b08788ff802109aa2ee64caf7a389cd4[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sun Dec 15 09:57:42 2019 +0530

    refactor PyG implementation

[33mcommit 077bf1ce3f12970e318647279fa52ff834cc4e83[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Sat Dec 14 17:51:23 2019 +0530

    remove redundant

[33mcommit 4fc753063f153de21b541a419c5140a664c271c1[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Thu Dec 12 23:18:49 2019 +0530

    restructured dir

[33mcommit acd42ac57ee7953457d874fbccd08ab61d1c39aa[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Thu Dec 12 23:12:20 2019 +0530

    updated blog+code

[33mcommit 91fc502001002e64ee53c7153f8f66a3f3260f9b[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Thu Dec 5 23:57:00 2019 +0530

    added nlp power law

[33mcommit f579458aebabc37e1e446c20c1998b666abd220f[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Thu Dec 5 23:46:36 2019 +0530

    added power law

[33mcommit caf550ca709b0b3b3a17741e1af53112ae9d7020[m
Author: shubham <shubham22019@gmail.com>
Date:   Sun Dec 1 23:05:37 2019 +0530

    added inductive_capability

[33mcommit e502732f197019a86b60636b6375459a3231f180[m
Author: shubham <shubham22019@gmail.com>
Date:   Sat Nov 30 22:46:42 2019 +0530

    adding aggregators

[33mcommit 2e395a04e8ea4326c965cc9e18547ce841ee3613[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Nov 30 19:44:41 2019 +0530

    Add files via upload

[33mcommit 054799622bde3657f0da5a7520bcc9aa82a6cda4[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Sat Nov 30 19:43:23 2019 +0530

    Add files via upload

[33mcommit 3d85b70ccbda54ba4345989a79184b72721f38d1[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Fri Nov 29 19:39:39 2019 +0530

    Add files via upload

[33mcommit ef82e2a06c2df1bb97382f81d4b78976b2ccd8b6[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Fri Nov 29 19:38:51 2019 +0530

    Create README.md

[33mcommit 24c73c9dfa71b6f40dd72125139e607086d86620[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Fri Nov 29 19:38:05 2019 +0530

    Add files via upload

[33mcommit 7ef4d58d2686e258324f3111377797823b8b7e26[m
Author: shubham <shubham22019@gmail.com>
Date:   Fri Nov 29 19:12:11 2019 +0530

    initial commit

[33mcommit 574d6aa21f84e022a1d0e866dae178b1405bb816[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Thu Nov 28 22:37:31 2019 +0530

    update blog

[33mcommit 6211aea74f24e92d984141a4e5ce363779cd224b[m
Author: Anirudh Dagar <anirudhdagar6@gmail.com>
Date:   Thu Oct 24 23:32:19 2019 +0530

    blog changes

[33mcommit 3cfaf9b26fa03c59f66c8852d07ec325b719f158[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Oct 15 22:40:58 2019 +0530

    Rename de1.ipynb to blog.ipynb

[33mcommit 2fad5ddc1a42a7c584539264cef4125e2c56fca3[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Oct 15 22:40:30 2019 +0530

    Prefinal blog

[33mcommit 870bc069de0b15393a9efe930119cb4bb7d2ef98[m
Author: shubhamgit1 <shubham22019@gmail.com>
Date:   Wed Sep 25 18:52:27 2019 +0530

    removing deepwalk

[33mcommit 3f0c1ccb69630666e55b11a1704c54fc9d54dcaa[m
Merge: 6baa011 8b2710c
Author: shubhamgit1 <shubham22019@gmail.com>
Date:   Wed Sep 25 18:13:26 2019 +0530

    Merge branch 'deepWalk' of https://github.com/dsgiitr/graph_nets into deepWalkNew

[33mcommit 6baa01113fe87dd9b88fd777793ba74d91b4c627[m
Author: shubhamgit1 <shubham22019@gmail.com>
Date:   Wed Sep 25 18:12:03 2019 +0530

    skip gram and random walk

[33mcommit 8b2710ce82ad06b3dc514e9ca78d071ecc25e528[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Wed Sep 25 14:35:52 2019 +0530

    Add files via upload

[33mcommit 9f61b37330f059acd13dfef19bda8e296794371c[m
Author: shubhamgit1 <shubham22019@gmail.com>
Date:   Wed Sep 25 13:17:22 2019 +0530

    random walk and skip gram

[33mcommit 49c7b865dccb7a6d9a4a6d25e2bd944e74c982e6[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Thu Aug 29 08:04:19 2019 +0530

    Add files via upload

[33mcommit 82a8c941cdd2a6246aa7082e489fc2d83b5dc493[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Thu Aug 29 08:03:58 2019 +0530

    Delete GCN_conv.ipynb

[33mcommit b40c18dc4b5800c55ad37ef39d8e88f3b8d83a1f[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Aug 27 04:11:03 2019 +0530

    Add files via upload

[33mcommit 93f7f9a0f59b95249ab6cf40df48bd91f2d133c3[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Aug 27 04:10:49 2019 +0530

    Delete gcn_conv_pytorch.ipynb

[33mcommit 7fcaafe29320e3aec14fe802d555a8ca3857b8b5[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Tue Aug 27 04:08:10 2019 +0530

    Add files via upload

[33mcommit 7d93f16b58179138bc80d20188320a512c95e651[m
Author: Ajit Pant <43715439+AjitPant@users.noreply.github.com>
Date:   Mon Aug 26 23:34:29 2019 +0530

    Add files via upload

[33mcommit 92e304987e050c8a1ccd7d6b7d922b7bff2a160b[m
Merge: 4431400 e3ea5f2
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Wed Aug 14 12:54:42 2019 +0530

    Merge pull request #2 from dsgiitr/revert-1-gcn_conv
    
    Revert "implementation on Cora dataset"

[33mcommit e3ea5f23c3a116060ce3b08145bd738e330005e2[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Wed Aug 14 12:54:07 2019 +0530

    Revert "implementation on Cora dataset"

[33mcommit 4431400b64541c6610df4df543ef9e564ad5b52a[m
Merge: 58e5752 c0d373a
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Wed Aug 14 12:53:07 2019 +0530

    Merge pull request #1 from dsgiitr/gcn_conv
    
    implementation on Cora dataset

[33mcommit c0d373a0ac754cc2533bf1e217c607c5727755be[m
Author: Shashank Gupta <47393639+gupta1912@users.noreply.github.com>
Date:   Wed Aug 14 12:52:37 2019 +0530

    implementation on Cora dataset

[33mcommit 58e57523474b968e1b81d957fa2312b3b01af7c1[m
Author: Anirudh <anirudhdagar6@gmail.com>
Date:   Wed Aug 7 17:03:15 2019 +0530

    Initial commit
