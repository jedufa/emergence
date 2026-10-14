/*
 * Bounded CUDA port of glibc 2.43 FMA double sin/cos/log arithmetic.
 *
 * Derived from GNU C Library files s_sin.c, sincostab.c, usncs.h,
 * e_log.c, and e_log_data.c (Copyright 2001-2026 Free Software
 * Foundation, Inc.).  SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * The port is deliberately narrower than general libm: Option D only calls
 * sin/cos for x in [0, 2*pi) and log for x in [2^-53, 1].  Explicit IEEE-754
 * binary64 FMA operations reproduce the current x86_64 glibc FMA dispatch;
 * surrounding CPython operations remain unfused.
 *
 * Source SHA-256:
 *   s_sin.c      874fe9ebf6accaeaf9f52df2de17c07925fb8ba133ce3e606e2b05f62ab1df91
 *   sincostab.c  a3ed5716ecbe5dfee173bfe6bad10663c9d239d5b01659d1fa834b3c97c3f355
 *   usncs.h      867fab124286036e22e760250ae929f8f1dcd9e6719cf4c40c8accf931abe749
 *   e_log.c      f61567475ce7ceb69b2d7f523a3f2c1eec4a3bc104057187dffe95c07180aeaf
 *   e_log_data.c 47340d52f647c947c875c3da300d0916f24200cd9949cabb912eb2e5f62bd003
 */

#pragma once

#include <cstdint>


static __device__ __constant__ uint64_t kOptionDGlibcSinCosBits[440] = {
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x3ff0000000000000ULL, 0x0000000000000000ULL,
    0x3f7fffeaaaaeeeefULL, 0xbc1e45e2ec67b77cULL, 0x3fefffc000155552ULL, 0x3c8f4a01a0196daeULL,
    0x3f8fffaaaaeeeed5ULL, 0xbc02ab639a9f0777ULL, 0x3fefff000155549fULL, 0x3c828a28a03a5ef3ULL,
    0x3f97ff7001033255ULL, 0x3bfefe2b51527336ULL, 0x3feffdc006bff7e6ULL, 0x3c8ae6dae86977bdULL,
    0x3f9ffeaaaeeee86fULL, 0xbc3cd406fb224ae2ULL, 0x3feffc00155527d3ULL, 0xbc83b54492d89b5bULL,
    0x3fa3feb2b12d45d5ULL, 0x3c34ec54203d1c11ULL, 0x3feff9c03414a7baULL, 0x3c6991f4be6c59bfULL,
    0x3fa7fdc01032fba9ULL, 0xbc4599bdf46e997aULL, 0x3feff7006bfdf99fULL, 0xbc78b3b560648d5fULL,
    0x3fabfc6d78586dacULL, 0x3c18e4fd03dbf236ULL, 0x3feff3c0c8103a31ULL, 0x3c74856dbddc0e66ULL,
    0x3faffaaaeeed4edbULL, 0xbc42d16d32684b69ULL, 0x3feff0015549f4d3ULL, 0x3c8328387b99426fULL,
    0x3fb1fc343d808befULL, 0xbc5f3d32e6f3be4fULL, 0x3fefebc222a8ef9fULL, 0x3c57934934f54c77ULL,
    0x3fb3facb12d1755bULL, 0xbc5921915299468cULL, 0x3fefe7034129ef6fULL, 0xbc6cbf4337c96f97ULL,
    0x3fb5f911fd10b737ULL, 0xbc50184f02be9102ULL, 0x3fefe1c4c3c873ebULL, 0xbc35a9c9057c4a02ULL,
    0x3fb7f701032550e4ULL, 0x3c3afc2d1800501aULL, 0x3fefdc06bf7e6b9bULL, 0x3c831902b535f8dbULL,
    0x3fb9f4902d55d1f9ULL, 0x3c52696d7eac1dc1ULL, 0x3fefd5c94b43e000ULL, 0xbc62e768cb4f92f9ULL,
    0x3fbbf1b78568391dULL, 0x3c5e91841dea4cc8ULL, 0x3fefcf0c800e99b1ULL, 0x3c6ea3d786d186acULL,
    0x3fbdee6f16c1cce6ULL, 0xbc450f8e2fb71673ULL, 0x3fefc7d078d1bc88ULL, 0x3c8075d2447db685ULL,
    0x3fbfeaaeee86ee36ULL, 0xbc4afcb2bcc6f03bULL, 0x3fefc015527d5bd3ULL, 0x3c8b68f35094efb8ULL,
    0x3fc0f3378ddd71d1ULL, 0x3c6d8468724f0f9eULL, 0x3fefb7db2bfe0695ULL, 0x3c821dadf4f65ab1ULL,
    0x3fc1f0d3d7afceafULL, 0xbc66ef95099769a5ULL, 0x3fefaf22263c4bd3ULL, 0xbc552ace133a2769ULL,
    0x3fc2ee285e4ab88fULL, 0xbc6e4d0f05dee058ULL, 0x3fefa5ea641c36f2ULL, 0x3c404da6ed17cc7cULL,
    0x3fc3eb312c5d66cbULL, 0x3c647d666b66cb91ULL, 0x3fef9c340a7cc428ULL, 0x3c8c5b6b063b7462ULL,
    0x3fc4e7ea4dc5f27bULL, 0x3c5949db2ac072fcULL, 0x3fef91ff40374d01ULL, 0xbc67d03f4d3a9e4cULL,
    0x3fc5e44fcfa126f3ULL, 0xbc66f443063f89b6ULL, 0x3fef874c2e1eecf6ULL, 0xbc8c6514e1332b16ULL,
    0x3fc6e05dc05a4d4cULL, 0xbbd32c5c8b81c940ULL, 0x3fef7c1afeffde24ULL, 0xbc78f55bc47540b1ULL,
    0x3fc7dc102fbaf2b5ULL, 0x3c45ab50e23c97c3ULL, 0x3fef706bdf9ece1cULL, 0xbc8698c80c36dcb4ULL,
    0x3fc8d7632efaa944ULL, 0xbc620fa262cbb953ULL, 0x3fef643efeb82acdULL, 0x3c76b00ac1fe28acULL,
    0x3fc9d252d0cec312ULL, 0x3c59c43d80b1137dULL, 0x3fef57948cff6797ULL, 0x3c6e3a0d3e03b1d5ULL,
    0x3fcaccdb297a0765ULL, 0xbc59883b57d6cdebULL, 0x3fef4a6cbd1e3a79ULL, 0x3c813df0edaebb57ULL,
    0x3fcbc6f84edc6199ULL, 0x3c69c1a56a7b0cabULL, 0x3fef3cc7c3b3d16eULL, 0xbc621a3ad28a3494ULL,
    0x3fccc0a6588289a3ULL, 0xbc6868d09bc87c6bULL, 0x3fef2ea5d753ffedULL, 0x3c8cc4215f56d583ULL,
    0x3fcdb9e15fb5a5d0ULL, 0xbc632e20d6cc6fc2ULL, 0x3fef20073086649fULL, 0x3c7b940416c1984bULL,
    0x3fceb2a57f8ae5a3ULL, 0xbc60be06af572cebULL, 0x3fef10ec09c5873bULL, 0x3c8d9072762c1283ULL,
    0x3fcfaaeed4f31577ULL, 0xbc615d88508e32b8ULL, 0x3fef01549f7deea1ULL, 0x3c8d3c1e99e5cafdULL,
    0x3fd0515cbf65155cULL, 0xbc79b8c29dfd8ec8ULL, 0x3feef141300d2f26ULL, 0xbc82aa1b08ded372ULL,
    0x3fd0cd00cef36436ULL, 0xbc79fb0a0c93e2b5ULL, 0x3feee0b1fbc0f11cULL, 0xbc4bfd2380bbc3b1ULL,
    0x3fd14861aa94ddebULL, 0xbc6be881b5b615a4ULL, 0x3feecfa744d5efa1ULL, 0xbc556d0a4af541d0ULL,
    0x3fd1c37d64c6b876ULL, 0x3c746076fe0dcff5ULL, 0x3feebe214f76efa8ULL, 0xbc802f9f12ba543eULL,
    0x3fd23e52111aaf36ULL, 0xbc74f080334eff18ULL, 0x3feeac2061bbaf4fULL, 0x3c62c1d53e94658dULL,
    0x3fd2b8ddc43eb49fULL, 0x3c61553899f2d807ULL, 0x3fee99a4c3a7cd83ULL, 0xbc82264b1bc53ce8ULL,
    0x3fd3331e94049f87ULL, 0x3c7e0cb6b40c302cULL, 0x3fee86aebf29a9edULL, 0x3c89397afdbb58a7ULL,
    0x3fd3ad129769d3d8ULL, 0x3c003d5504878398ULL, 0x3fee733ea0193d40ULL, 0xbc86428b3546ce13ULL,
    0x3fd426b7e69ee697ULL, 0xbc7f09c75705c59fULL, 0x3fee5f54b436e9d0ULL, 0x3c87eb0fd02fc8bcULL,
    0x3fd4a00c9b0f3d20ULL, 0x3c7823ba6bb08eadULL, 0x3fee4af14b2a449cULL, 0xbc868ca02e8a6833ULL,
    0x3fd5190ecf68a77aULL, 0x3c7b357155eef0f3ULL, 0x3fee3614b680d6a5ULL, 0xbc727793aa015237ULL,
    0x3fd591bc9fa2f597ULL, 0x3c67c74bac3fe0cbULL, 0x3fee20bf49acd6c1ULL, 0xbc5660aec7ef636cULL,
    0x3fd60a1429078775ULL, 0x3c5b1fd80ba89133ULL, 0x3fee0af15a03dbceULL, 0x3c5fe8e702771ae6ULL,
    0x3fd682138a38d7f7ULL, 0xbc7d889202444aadULL, 0x3fedf4ab3ebd875eULL, 0xbc8e2d8a7e6736c4ULL,
    0x3fd6f9b8e33a0255ULL, 0x3c742bc14ee9da0dULL, 0x3feddded50f228d6ULL, 0xbc6e80c8d42ba2bfULL,
    0x3fd7710255764214ULL, 0xbc66ead7314bb6ceULL, 0x3fedc6b7eb995912ULL, 0x3c54b364776dcd35ULL,
    0x3fd7e7ee03c86d4eULL, 0xbc7b63bcdabf5af2ULL, 0x3fedaf0b6b888e83ULL, 0x3c8a249e2b5e5ceaULL,
    0x3fd85e7a12826949ULL, 0x3c78a40e9b5face0ULL, 0x3fed96e82f71a9dcULL, 0x3c8ff61bd5d2039dULL,
    0x3fd8d4a4a774992fULL, 0x3c744a02ea766326ULL, 0x3fed7e4e97e17b4aULL, 0xbc63b770352bed94ULL,
    0x3fd94a6be9f546c5ULL, 0xbc769ce13e683f58ULL, 0x3fed653f073e4040ULL, 0xbc876236434bec37ULL,
    0x3fd9bfce02e80510ULL, 0x3c709e39a320b0a4ULL, 0x3fed4bb9e1c619e0ULL, 0x3c8f34bb77858f61ULL,
    0x3fda34c91cc50ccaULL, 0xbc5a310e3b50cecdULL, 0x3fed31bf8d8d7c06ULL, 0x3c7e60dd3089cbddULL,
    0x3fdaa95b63a09277ULL, 0xbc66293eb13c0381ULL, 0x3fed1750727d94f0ULL, 0x3c80d52b1ec1a48eULL,
    0x3fdb1d8305321617ULL, 0xbc7ae242cb99f519ULL, 0x3fecfc6cfa52ad9fULL, 0x3c88b5b5508f2a0dULL,
    0x3fdb913e30dbac43ULL, 0xbc7e38ad2f6c3ff1ULL, 0x3fece115909a82e5ULL, 0x3c81f139bb31109aULL,
    0x3fdc048b17b140a3ULL, 0x3c619fe6757e9fa7ULL, 0x3fecc54aa2b2972eULL, 0x3c64ee162ba83a98ULL,
    0x3fdc7767ec7fd19eULL, 0xbc5eb14d1a3d5826ULL, 0x3feca90c9fc67d0bULL, 0xbc646a81485e3462ULL,
    0x3fdce9d2e3d4a51fULL, 0xbc62fc8a12dae298ULL, 0x3fec8c5bf8ce1a84ULL, 0x3c7ab3d1a1590123ULL,
    0x3fdd5bca34047661ULL, 0x3c728a44a75fc29cULL, 0x3fec6f39208be53bULL, 0xbc8741dbfbaadb42ULL,
    0x3fddcd4c15329c9aULL, 0x3c70d4c6e171fd9aULL, 0x3fec51a48b8b175eULL, 0xbc61bbb43b9aa880ULL,
    0x3fde3e56c1582a69ULL, 0xbc50a4821099f88fULL, 0x3fec339eb01ddd81ULL, 0xbc8caaf5ee82c5c0ULL,
    0x3fdeaee8744b05f0ULL, 0xbc5789b43c9b027dULL, 0x3fec1528065b7d50ULL, 0xbc8892111312e828ULL,
    0x3fdf1eff6bc4f97bULL, 0x3c717212f8a7525cULL, 0x3febf641081e7536ULL, 0x3c8b7bd71628a9a1ULL,
    0x3fdf8e99e76abc97ULL, 0x3c59d950af2d00a3ULL, 0x3febd6ea310294f5ULL, 0x3c731bbcc88c109dULL,
    0x3fdffdb628d2f57aULL, 0x3c6f4a992e905b6aULL, 0x3febb723fe630f32ULL, 0x3c772bd2452d0a39ULL,
    0x3fe0362939c69955ULL, 0xbc82d8cd78397b01ULL, 0x3feb96eeef58840eULL, 0x3c545a3cc78fade0ULL,
    0x3fe06d3686946e5bULL, 0x3c83f5ae4538ff1bULL, 0x3feb764b84b704c2ULL, 0xbc8f5848c21b389bULL,
    0x3fe0a4021e9e1001ULL, 0xbc86f643a13914f6ULL, 0x3feb553a410c104eULL, 0x3c58ff7947027a16ULL,
    0x3fe0da8b26b5672eULL, 0xbc8a58def0bee909ULL, 0x3feb33bba89c8948ULL, 0x3c8ea6a51d1f6ca9ULL,
    0x3fe110d0c4b69c3bULL, 0x3c8d918998809981ULL, 0x3feb11d04162a4c6ULL, 0x3c71dd561efbc0c2ULL,
    0x3fe146d21f8b7f82ULL, 0x3c7bf9535e2739a8ULL, 0x3feaef78930bd275ULL, 0xbc7f836279746f94ULL,
    0x3fe17c8e5f2eedb0ULL, 0x3c635e57102e2488ULL, 0x3feaccb526f69de5ULL, 0x3c88fb6a8dd6b6ccULL,
    0x3fe1b204acb02fddULL, 0xbc5f190c70cbb5ffULL, 0x3feaa98688308913ULL, 0xbc0b83d607cd5070ULL,
    0x3fe1e7343236574cULL, 0x3c722a3fa4f41d5aULL, 0x3fea85ed4373e02dULL, 0x3c69be06385ec792ULL,
    0x3fe21c1c1b0394cfULL, 0x3c5e5b324b23aa31ULL, 0x3fea61e9e72586afULL, 0x3c858330e2fd453fULL,
    0x3fe250bb93788bbbULL, 0x3c7ea3d02457bcceULL, 0x3fea3d7d0352bdcfULL, 0xbc868dbaeca19669ULL,
    0x3fe28511c917a067ULL, 0xbc801df1d9a16b70ULL, 0x3fea18a729aee445ULL, 0x3c395e25736c0358ULL,
    0x3fe2b91dea88421eULL, 0xbc8fa371db216ab0ULL, 0x3fe9f368ed912f85ULL, 0xbc81d200c5791606ULL,
    0x3fe2ecdf279a3082ULL, 0x3c8d3557e0e7e37eULL, 0x3fe9cdc2e3f25e5cULL, 0x3c83f99112993f62ULL,
    0x3fe32054b148bc4fULL, 0x3c8f6b42095a135bULL, 0x3fe9a7b5a36a6514ULL, 0x3c8722cfcc9fa7a9ULL,
    0x3fe3537db9be0367ULL, 0x3c6b327e7af040f0ULL, 0x3fe98141c42e1310ULL, 0x3c8d1ff80488f08dULL,
    0x3fe386597456282bULL, 0xbc710fada93b07a8ULL, 0x3fe95a67e00cb1fdULL, 0xbc80befda21f862dULL,
    0x3fe3b8e715a2840aULL, 0xbc797653a7d2f07bULL, 0x3fe93328926d9e92ULL, 0xbc8bb77003600cdaULL,
    0x3fe3eb25d36cd53aULL, 0xbc5be570e1570fc0ULL, 0x3fe90b84784ddaf7ULL, 0xbc70feb10ab93b87ULL,
    0x3fe41d14e4ba6790ULL, 0x3c84608fd287ecf5ULL, 0x3fe8e37c303d9ad1ULL, 0xbc6463a4b53d4bf8ULL,
    0x3fe44eb381cf386bULL, 0xbc83ed6c1e6a5505ULL, 0x3fe8bb105a5dc900ULL, 0x3c8863e03e9474c1ULL,
    0x3fe48000e431159fULL, 0xbc8b194a7463ed10ULL, 0x3fe89241985d871fULL, 0x3c8c48d9c413ed84ULL,
    0x3fe4b0fc46aab761ULL, 0x3c20da05738cc59aULL, 0x3fe869108d77a6c6ULL, 0x3c7338ffe2bfe9ddULL,
    0x3fe4e1a4e54ed51bULL, 0xbc8a492f89b7c76aULL, 0x3fe83f7dde701ca0ULL, 0xbc4152cf609bc6e8ULL,
    0x3fe511f9fd7b351cULL, 0xbc85c0e861c48831ULL, 0x3fe8158a31916d5dULL, 0xbc6de8b90b8228deULL,
    0x3fe541facddbb724ULL, 0x3c7232c28520d391ULL, 0x3fe7eb362eaa1488ULL, 0x3c5a1d65a4a5959fULL,
    0x3fe571a6966d59b3ULL, 0x3c5c843b4d0fb198ULL, 0x3fe7c0827f09e54fULL, 0xbc6c73d6d72aee68ULL,
    0x3fe5a0fc98813a12ULL, 0xbc8d82e2b7d4227bULL, 0x3fe7956fcd7f6543ULL, 0xbc8ab276e9d45ae4ULL,
    0x3fe5cffc16bf8f0dULL, 0x3c896cb370eb578aULL, 0x3fe769fec655211fULL, 0xbc6827d5cf8c68c5ULL,
    0x3fe5fea4552a9e57ULL, 0x3c80b6cef7ee20b7ULL, 0x3fe73e30174efba1ULL, 0xbc65d3ae3d94ad5fULL,
    0x3fe62cf49921ac79ULL, 0xbc8edd9855b6241aULL, 0x3fe712046fa77678ULL, 0x3c8425b0a5029c81ULL,
    0x3fe65aec2963e755ULL, 0x3c8126f96b71053cULL, 0x3fe6e57c800cf55eULL, 0x3c860286dedbd0a6ULL,
    0x3fe6888a4e134b2fULL, 0xbc86b7d37644d5e6ULL, 0x3fe6b898fa9efb5dULL, 0x3c715ac786ccf4b2ULL,
    0x3fe6b5ce50b7821aULL, 0xbc65d5158f702e0fULL, 0x3fe68b5a92eb6253ULL, 0xbc89a91ad985f89cULL,
    0x3fe6e2b77c40bde1ULL, 0xbc70e729857fad53ULL, 0x3fe65dc1fdeb8cbaULL, 0xbc597c1b47337c77ULL,
    0x3fe70f451d0a8c40ULL, 0x3c697ede3885770dULL, 0x3fe62fcff20191c7ULL, 0x3c6d9143895756efULL,
    0x3fe73b7680dea578ULL, 0xbc72248306dc12a2ULL, 0x3fe6018526f563dfULL, 0x3c846ca5e0e432d0ULL,
    0x3fe7674af6f7b524ULL, 0x3c7e9d3f94ac84a8ULL, 0x3fe5d2e255f1f17aULL, 0x3c80314104c8892bULL,
    0x3fe792c1d0041d52ULL, 0xbc8abf05eeb354ebULL, 0x3fe5a3e839824077ULL, 0x3c8428aa2759be62ULL,
    0x3fe7bdda5e28b3c2ULL, 0x3c4ad1197ccd0393ULL, 0x3fe574978d8e83f2ULL, 0x3c8f4714af282d23ULL,
    0x3fe7e893f5037959ULL, 0x3c80eefbaa650c4cULL, 0x3fe544f10f592ca5ULL, 0xbc8e7ae8e6c7a62fULL,
    0x3fe812ede9ae4ba4ULL, 0xbc87830adf402ddaULL, 0x3fe514f57d7bf3daULL, 0x3c747a108073c259ULL,
};

static __device__ __constant__ uint64_t kOptionDGlibcLogPoly1Bits[11] = {
    0xbfe0000000000000ULL, 0x3fd5555555555577ULL, 0xbfcffffffffffdcbULL, 0x3fc999999995dd0cULL,
    0xbfc55555556745a7ULL, 0x3fc24924a344de30ULL, 0xbfbfffffa4423d65ULL, 0x3fbc7184282ad6caULL,
    0xbfb999eb43b068ffULL, 0x3fb78182f7afd085ULL, 0xbfb5521375d145cdULL,
};

static __device__ __constant__ uint64_t kOptionDGlibcLogPolyBits[5] = {
    0xbfe0000000000001ULL, 0x3fd555555551305bULL, 0xbfcfffffffeb4590ULL, 0x3fc999b324f10111ULL,
    0xbfc55575e506c89fULL,
};

static __device__ __constant__ uint64_t kOptionDGlibcLogTableBits[256] = {
    0x3ff734f0c3e0de9fULL, 0xbfd7cc7f79e69000ULL, 0x3ff713786a2ce91fULL, 0xbfd76feec20d0000ULL,
    0x3ff6f26008fab5a0ULL, 0xbfd713e31351e000ULL, 0x3ff6d1a61f138c7dULL, 0xbfd6b85b38287800ULL,
    0x3ff6b1490bc5b4d1ULL, 0xbfd65d5590807800ULL, 0x3ff69147332f0cbaULL, 0xbfd602d076180000ULL,
    0x3ff6719f18224223ULL, 0xbfd5a8ca86909000ULL, 0x3ff6524f99a51ed9ULL, 0xbfd54f4356035000ULL,
    0x3ff63356aa8f24c4ULL, 0xbfd4f637c36b4000ULL, 0x3ff614b36b9ddc14ULL, 0xbfd49da7fda85000ULL,
    0x3ff5f66452c65c4cULL, 0xbfd445923989a800ULL, 0x3ff5d867b5912c4fULL, 0xbfd3edf439b0b800ULL,
    0x3ff5babccb5b90deULL, 0xbfd396ce448f7000ULL, 0x3ff59d61f2d91a78ULL, 0xbfd3401e17bda000ULL,
    0x3ff5805612465687ULL, 0xbfd2e9e2ef468000ULL, 0x3ff56397cee76bd3ULL, 0xbfd2941b3830e000ULL,
    0x3ff54725e2a77f93ULL, 0xbfd23ec58cda8800ULL, 0x3ff52aff42064583ULL, 0xbfd1e9e129279000ULL,
    0x3ff50f22dbb2bddfULL, 0xbfd1956d2b48f800ULL, 0x3ff4f38f4734ded7ULL, 0xbfd141679ab9f800ULL,
    0x3ff4d843cfde2840ULL, 0xbfd0edd094ef9800ULL, 0x3ff4bd3ec078a3c8ULL, 0xbfd09aa518db1000ULL,
    0x3ff4a27fc3e0258aULL, 0xbfd047e65263b800ULL, 0x3ff4880524d48434ULL, 0xbfcfeb224586f000ULL,
    0x3ff46dce1b192d0bULL, 0xbfcf474a7517b000ULL, 0x3ff453d9d3391854ULL, 0xbfcea4443d103000ULL,
    0x3ff43a2744b4845aULL, 0xbfce020d44e9b000ULL, 0x3ff420b54115f8fbULL, 0xbfcd60a22977f000ULL,
    0x3ff40782da3ef4b1ULL, 0xbfccc00104959000ULL, 0x3ff3ee8f5d57fe8fULL, 0xbfcc202956891000ULL,
    0x3ff3d5d9a00b4ce9ULL, 0xbfcb81178d811000ULL, 0x3ff3bd60c010c12bULL, 0xbfcae2c9ccd3d000ULL,
    0x3ff3a5242b75dab8ULL, 0xbfca45402e129000ULL, 0x3ff38d22cd9fd002ULL, 0xbfc9a877681df000ULL,
    0x3ff3755bc5847a1cULL, 0xbfc90c6d69483000ULL, 0x3ff35dce49ad36e2ULL, 0xbfc87120a645c000ULL,
    0x3ff34679984dd440ULL, 0xbfc7d68fb4143000ULL, 0x3ff32f5cceffcb24ULL, 0xbfc73cb83c627000ULL,
    0x3ff3187775a10d49ULL, 0xbfc6a39a9b376000ULL, 0x3ff301c8373e3990ULL, 0xbfc60b3154b7a000ULL,
    0x3ff2eb4ebb95f841ULL, 0xbfc5737d76243000ULL, 0x3ff2d50a0219a9d1ULL, 0xbfc4dc7b8fc23000ULL,
    0x3ff2bef9a8b7fd2aULL, 0xbfc4462c51d20000ULL, 0x3ff2a91c7a0c1babULL, 0xbfc3b08abc830000ULL,
    0x3ff293726014b530ULL, 0xbfc31b996b490000ULL, 0x3ff27dfa5757a1f5ULL, 0xbfc2875490a44000ULL,
    0x3ff268b39b1d3bbfULL, 0xbfc1f3b9f879a000ULL, 0x3ff2539d838ff5bdULL, 0xbfc160c8252ca000ULL,
    0x3ff23eb7aac9083bULL, 0xbfc0ce7f57f72000ULL, 0x3ff22a012ba940b6ULL, 0xbfc03cdc49fea000ULL,
    0x3ff2157996cc4132ULL, 0xbfbf57bdbc4b8000ULL, 0x3ff201201dd2fc9bULL, 0xbfbe370896404000ULL,
    0x3ff1ecf4494d480bULL, 0xbfbd17983ef94000ULL, 0x3ff1d8f5528f6569ULL, 0xbfbbf9674ed8a000ULL,
    0x3ff1c52311577e7cULL, 0xbfbadc79202f6000ULL, 0x3ff1b17c74cb26e9ULL, 0xbfb9c0c3e7288000ULL,
    0x3ff19e010c2c1ab6ULL, 0xbfb8a646b372c000ULL, 0x3ff18ab07bb670bdULL, 0xbfb78d01b3ac0000ULL,
    0x3ff1778a25efbcb6ULL, 0xbfb674f145380000ULL, 0x3ff1648d354c31daULL, 0xbfb55e0e6d878000ULL,
    0x3ff151b990275fddULL, 0xbfb4485cdea1e000ULL, 0x3ff13f0ea432d24cULL, 0xbfb333d94d6aa000ULL,
    0x3ff12c8b7210f9daULL, 0xbfb22079f8c56000ULL, 0x3ff11a3028ecb531ULL, 0xbfb10e4698622000ULL,
    0x3ff107fbda8434afULL, 0xbfaffa6c6ad20000ULL, 0x3ff0f5ee0f4e6bb3ULL, 0xbfadda8d4a774000ULL,
    0x3ff0e4065d2a9fceULL, 0xbfabbcece4850000ULL, 0x3ff0d244632ca521ULL, 0xbfa9a1894012c000ULL,
    0x3ff0c0a77ce2981aULL, 0xbfa788583302c000ULL, 0x3ff0af2f83c636d1ULL, 0xbfa5715e67d68000ULL,
    0x3ff09ddb98a01339ULL, 0xbfa35c8a49658000ULL, 0x3ff08cabaf52e7dfULL, 0xbfa149e364154000ULL,
    0x3ff07b9f2f4e28fbULL, 0xbf9e72c082eb8000ULL, 0x3ff06ab58c358f19ULL, 0xbf9a55f152528000ULL,
    0x3ff059eea5ecf92cULL, 0xbf963d62cf818000ULL, 0x3ff04949cdd12c90ULL, 0xbf9228fb8caa0000ULL,
    0x3ff038c6c6f0ada9ULL, 0xbf8c317b20f90000ULL, 0x3ff02865137932a9ULL, 0xbf8419355daa0000ULL,
    0x3ff0182427ea7348ULL, 0xbf781203c2ec0000ULL, 0x3ff008040614b195ULL, 0xbf60040979240000ULL,
    0x3fefe01ff726fa1aULL, 0x3f6feff384900000ULL, 0x3fefa11cc261ea74ULL, 0x3f87dc41353d0000ULL,
    0x3fef6310b081992eULL, 0x3f93cea3c4c28000ULL, 0x3fef25f63ceeadcdULL, 0x3f9b9fc114890000ULL,
    0x3feee9c8039113e7ULL, 0x3fa1b0d8ce110000ULL, 0x3feeae8078cbb1abULL, 0x3fa58a5bd001c000ULL,
    0x3fee741aa29d0c9bULL, 0x3fa95c8340d88000ULL, 0x3fee3a91830a99b5ULL, 0x3fad276aef578000ULL,
    0x3fee01e009609a56ULL, 0x3fb07598e598c000ULL, 0x3fedca01e577bb98ULL, 0x3fb253f5e30d2000ULL,
    0x3fed92f20b7c9103ULL, 0x3fb42edd8b380000ULL, 0x3fed5cac66fb5cceULL, 0x3fb606598757c000ULL,
    0x3fed272caa5ede9dULL, 0x3fb7da76356a0000ULL, 0x3fecf26e3e6b2ccdULL, 0x3fb9ab434e1c6000ULL,
    0x3fecbe6da2a77902ULL, 0x3fbb78c7bb0d6000ULL, 0x3fec8b266d37086dULL, 0x3fbd431332e72000ULL,
    0x3fec5894bd5d5804ULL, 0x3fbf0a3171de6000ULL, 0x3fec26b533bb9f8cULL, 0x3fc067152b914000ULL,
    0x3febf583eeece73fULL, 0x3fc147858292b000ULL, 0x3febc4fd75db96c1ULL, 0x3fc2266ecdca3000ULL,
    0x3feb951e0c864a28ULL, 0x3fc303d7a6c55000ULL, 0x3feb65e2c5ef3e2cULL, 0x3fc3dfc33c331000ULL,
    0x3feb374867c9888bULL, 0x3fc4ba366b7a8000ULL, 0x3feb094b211d304aULL, 0x3fc5933928d1f000ULL,
    0x3feadbe885f2ef7eULL, 0x3fc66acd2418f000ULL, 0x3feaaf1d31603da2ULL, 0x3fc740f8ec669000ULL,
    0x3fea82e63fd358a7ULL, 0x3fc815c0f51af000ULL, 0x3fea5740ef09738bULL, 0x3fc8e92954f68000ULL,
    0x3fea2c2a90ab4b27ULL, 0x3fc9bb3602f84000ULL, 0x3fea01a01393f2d1ULL, 0x3fca8bed1c2c0000ULL,
    0x3fe9d79f24db3c1bULL, 0x3fcb5b515c01d000ULL, 0x3fe9ae2505c7b190ULL, 0x3fcc2967ccbcc000ULL,
    0x3fe9852ef297ce2fULL, 0x3fccf635d5486000ULL, 0x3fe95cbaeea44b75ULL, 0x3fcdc1bd3446c000ULL,
    0x3fe934c69de74838ULL, 0x3fce8c01b8cfe000ULL, 0x3fe90d4f2f6752e6ULL, 0x3fcf5509c0179000ULL,
    0x3fe8e6528effd79dULL, 0x3fd00e6c121fb800ULL, 0x3fe8bfce9fcc007cULL, 0x3fd071b80e93d000ULL,
    0x3fe899c0dabec30eULL, 0x3fd0d46b9e867000ULL, 0x3fe87427aa2317fbULL, 0x3fd13687334bd000ULL,
    0x3fe84f00acb39a08ULL, 0x3fd1980d67234800ULL, 0x3fe82a49e8653e55ULL, 0x3fd1f8ffe0cc8000ULL,
    0x3fe8060195f40260ULL, 0x3fd2595fd7636800ULL, 0x3fe7e22563e0a329ULL, 0x3fd2b9300914a800ULL,
    0x3fe7beb377dcb5adULL, 0x3fd3187210436000ULL, 0x3fe79baa679725c2ULL, 0x3fd377266dec1800ULL,
    0x3fe77907f2170657ULL, 0x3fd3d54ffbaf3000ULL, 0x3fe756cadbd6130cULL, 0x3fd432eee32fe000ULL,
};


__device__ __forceinline__ double option_d_glibc_bits_double(const uint64_t bits) {
    return __longlong_as_double(bits);
}


__device__ __forceinline__ uint64_t option_d_glibc_double_bits(const double value) {
    return __double_as_longlong(value);
}


__device__ __forceinline__ double option_d_glibc_abs(const double value) {
    return option_d_glibc_bits_double(
        option_d_glibc_double_bits(value) & UINT64_C(0x7fffffffffffffff)
    );
}


__device__ __forceinline__ double option_d_glibc_copysign(
    const double magnitude,
    const double sign
) {
    return option_d_glibc_bits_double(
        (option_d_glibc_double_bits(magnitude) & UINT64_C(0x7fffffffffffffff))
        | (option_d_glibc_double_bits(sign) & UINT64_C(0x8000000000000000))
    );
}


__device__ __forceinline__ double option_d_fma(
    const double left,
    const double right,
    const double addend
) {
    return __fma_rn(left, right, addend);
}


__device__ __forceinline__ double option_d_fnma(
    const double left,
    const double right,
    const double addend
) {
    return __fma_rn(-left, right, addend);
}


__device__ __forceinline__ double option_d_fms(
    const double left,
    const double right,
    const double subtrahend
) {
    return __fma_rn(left, right, -subtrahend);
}


__device__ __forceinline__ double option_d_sincos_table(const int32_t index) {
    return option_d_glibc_bits_double(kOptionDGlibcSinCosBits[index]);
}


__device__ __forceinline__ double option_d_do_cos(double x, double dx) {
    constexpr double big = 0x1.8000000000000p45;
    constexpr double sn3 = -1.66666666666664880952546298448555E-01;
    constexpr double sn5 = 8.33333214285722277379541354343671E-03;
    constexpr double cs2 = 4.99999999999999999999950396842453E-01;
    constexpr double cs4 = -4.16666666666664434524222570944589E-02;
    constexpr double cs6 = 1.38888874007937613028114285595617E-03;
    if (x < 0.0) {
        dx = -dx;
    }
    const double absolute = option_d_glibc_abs(x);
    const double quantized = big + absolute;
    x = absolute - (quantized - big) + dx;
    const double xx = x * x;
    const double sin_poly = option_d_fma(xx, sn5, sn3);
    const double s = option_d_fma(x * xx, sin_poly, x);
    const double cos_poly_inner = option_d_fma(xx, cs6, cs4);
    const double cos_poly = option_d_fma(xx, cos_poly_inner, cs2);
    const double c = xx * cos_poly;
    const int32_t index = static_cast<int32_t>(
        option_d_glibc_double_bits(quantized) & UINT64_C(0xffffffff)
    ) << 2;
    const double sn = option_d_sincos_table(index);
    const double ssn = option_d_sincos_table(index + 1);
    const double cs = option_d_sincos_table(index + 2);
    const double ccs = option_d_sincos_table(index + 3);
    double correction = option_d_fnma(s, ssn, ccs);
    correction = option_d_fnma(c, cs, correction);
    correction = option_d_fnma(s, sn, correction);
    return cs + correction;
}


__device__ __forceinline__ double option_d_taylor_sin(
    const double x,
    const double dx
) {
    constexpr double s1 = -0x1.5555555555555p-3;
    constexpr double s2 = 0x1.1111111110ECEp-7;
    constexpr double s3 = -0x1.A01A019DB08B8p-13;
    constexpr double s4 = 0x1.71DE27B9A7ED9p-19;
    constexpr double s5 = -0x1.ADDFFC2FCDF59p-26;
    const double xx = x * x;
    double polynomial = option_d_fma(xx, s5, s4);
    polynomial = option_d_fma(xx, polynomial, s3);
    polynomial = option_d_fma(xx, polynomial, s2);
    polynomial = option_d_fma(xx, polynomial, s1);
    const double half_dx = 0.5 * dx;
    const double product = option_d_fms(x, polynomial, half_dx);
    const double correction = option_d_fma(xx, product, dx);
    return x + correction;
}


__device__ __forceinline__ double option_d_do_sin(double x, double dx) {
    constexpr double big = 0x1.8000000000000p45;
    constexpr double sn3 = -1.66666666666664880952546298448555E-01;
    constexpr double sn5 = 8.33333214285722277379541354343671E-03;
    constexpr double cs2 = 4.99999999999999999999950396842453E-01;
    constexpr double cs4 = -4.16666666666664434524222570944589E-02;
    constexpr double cs6 = 1.38888874007937613028114285595617E-03;
    const double original = x;
    if (option_d_glibc_abs(x) < 0.126) {
        return option_d_taylor_sin(x, dx);
    }
    if (x <= 0.0) {
        dx = -dx;
    }
    const double absolute = option_d_glibc_abs(x);
    const double quantized = big + absolute;
    x = absolute - (quantized - big);
    const double xx = x * x;
    const double sin_poly = option_d_fma(xx, sn5, sn3);
    const double s = x + option_d_fma(x * xx, sin_poly, dx);
    const double cos_poly_inner = option_d_fma(xx, cs6, cs4);
    const double cos_poly = option_d_fma(xx, cos_poly_inner, cs2);
    const double c = option_d_fma(x, dx, xx * cos_poly);
    const int32_t index = static_cast<int32_t>(
        option_d_glibc_double_bits(quantized) & UINT64_C(0xffffffff)
    ) << 2;
    const double sn = option_d_sincos_table(index);
    const double ssn = option_d_sincos_table(index + 1);
    const double cs = option_d_sincos_table(index + 2);
    const double ccs = option_d_sincos_table(index + 3);
    double correction = option_d_fma(s, ccs, ssn);
    correction = option_d_fnma(c, sn, correction);
    correction = option_d_fma(s, cs, correction);
    return option_d_glibc_copysign(sn + correction, original);
}


__device__ __forceinline__ int32_t option_d_reduce_sincos(
    const double x,
    double* reduced,
    double* tail
) {
    constexpr double hpinv = 0x1.45F306DC9C883p-1;
    constexpr double toint = 0x1.8000000000000p52;
    constexpr double mp1 = 0x1.921FB58000000p0;
    constexpr double mp2 = -0x1.DDE973C000000p-27;
    constexpr double pp3 = -0x1.CB3B398000000p-55;
    constexpr double pp4 = -0x1.d747f23e32ed7p-83;
    const double rounded = option_d_fma(x, hpinv, toint);
    const double xn = rounded - toint;
    const double high = option_d_fnma(xn, mp1, x);
    const double y = option_d_fma(xn, -mp2, high);
    const int32_t quadrant = static_cast<int32_t>(
        option_d_glibc_double_bits(rounded) & UINT64_C(0xffffffff)
    ) & 3;
    const double second = option_d_fnma(xn, pp3, y);
    const double first_error = y - second;
    double low = option_d_fnma(xn, pp3, first_error);
    const double first = option_d_fnma(xn, pp4, second);
    const double second_error = second - first;
    low += option_d_fnma(xn, pp4, second_error);
    *reduced = first;
    *tail = low;
    return quadrant;
}


__device__ __forceinline__ double option_d_do_sincos(
    const double reduced,
    const double tail,
    const int32_t quadrant
) {
    const double value = (quadrant & 1)
        ? option_d_do_cos(reduced, tail)
        : option_d_do_sin(reduced, tail);
    return (quadrant & 2) ? -value : value;
}


__device__ __forceinline__ double option_d_glibc_sin_v1(const double x) {
    constexpr double hp0 = 0x1.921FB54442D18p0;
    constexpr double hp1 = 0x1.1A62633145C07p-54;
    const int32_t high = static_cast<int32_t>(
        option_d_glibc_double_bits(x) >> 32
    ) & 0x7fffffff;
    if (high < 0x3e500000) {
        return x;
    }
    if (high < 0x3feb6000) {
        return option_d_do_sin(x, 0.0);
    }
    if (high < 0x400368fd) {
        return option_d_glibc_copysign(
            option_d_do_cos(hp0 - option_d_glibc_abs(x), hp1),
            x
        );
    }
    double reduced;
    double tail;
    const int32_t quadrant = option_d_reduce_sincos(x, &reduced, &tail);
    return option_d_do_sincos(reduced, tail, quadrant);
}


__device__ __forceinline__ double option_d_glibc_cos_v1(const double x) {
    constexpr double hp0 = 0x1.921FB54442D18p0;
    constexpr double hp1 = 0x1.1A62633145C07p-54;
    const int32_t high = static_cast<int32_t>(
        option_d_glibc_double_bits(x) >> 32
    ) & 0x7fffffff;
    if (high < 0x3e400000) {
        return 1.0;
    }
    if (high < 0x3feb6000) {
        return option_d_do_cos(x, 0.0);
    }
    if (high < 0x400368fd) {
        const double y = hp0 - option_d_glibc_abs(x);
        const double reduced = y + hp1;
        const double tail = (y - reduced) + hp1;
        return option_d_do_sin(reduced, tail);
    }
    double reduced;
    double tail;
    const int32_t quadrant = option_d_reduce_sincos(x, &reduced, &tail);
    return option_d_do_sincos(reduced, tail, quadrant + 1);
}


__device__ __forceinline__ double option_d_log_poly1(const int32_t index) {
    return option_d_glibc_bits_double(kOptionDGlibcLogPoly1Bits[index]);
}


__device__ __forceinline__ double option_d_log_poly(const int32_t index) {
    return option_d_glibc_bits_double(kOptionDGlibcLogPolyBits[index]);
}


__device__ __forceinline__ double option_d_log_table(const int32_t index) {
    return option_d_glibc_bits_double(kOptionDGlibcLogTableBits[index]);
}


__device__ __forceinline__ double option_d_glibc_log_v1(const double x) {
    const uint64_t bits = option_d_glibc_double_bits(x);
    constexpr uint64_t lower = UINT64_C(0x3fee000000000000);
    constexpr uint64_t upper = UINT64_C(0x3ff1090000000000);
    if (bits - lower < upper - lower) {
        if (bits == UINT64_C(0x3ff0000000000000)) {
            return 0.0;
        }
        const double r = x - 1.0;
        const double r2 = r * r;
        const double r3 = r * r2;
        double first = option_d_fma(option_d_log_poly1(2), r, option_d_log_poly1(1));
        first = option_d_fma(option_d_log_poly1(3), r2, first);
        double second = option_d_fma(option_d_log_poly1(5), r, option_d_log_poly1(4));
        second = option_d_fma(option_d_log_poly1(6), r2, second);
        double third = option_d_fma(option_d_log_poly1(8), r, option_d_log_poly1(7));
        third = option_d_fma(option_d_log_poly1(9), r2, third);
        third = option_d_fma(option_d_log_poly1(10), r3, third);
        second = option_d_fma(third, r3, second);
        first = option_d_fma(second, r3, first);
        const double combined = option_d_fma(r, 0x1p27, r);
        const double rhi = option_d_fnma(r, 0x1p27, combined);
        const double rlo = r - rhi;
        const double square = rhi * rhi;
        const double coefficient = option_d_log_poly1(0);
        const double hi = option_d_fma(square, coefficient, r);
        double lo = option_d_fma(square, coefficient, r - hi);
        lo = option_d_fma(coefficient * rlo, rhi + r, lo);
        const double y = option_d_fma(first, r3, lo);
        return hi + y;
    }
    constexpr uint64_t offset = UINT64_C(0x3fe6000000000000);
    const uint64_t temporary = bits - offset;
    const int32_t index = static_cast<int32_t>((temporary >> 45) & 127);
    const int32_t exponent = static_cast<int32_t>(
        static_cast<int64_t>(temporary) >> 52
    );
    const uint64_t normalized_bits = bits
        - (temporary & (UINT64_C(0xfff) << 52));
    const double normalized = option_d_glibc_bits_double(normalized_bits);
    const double invc = option_d_log_table(2 * index);
    const double logc = option_d_log_table(2 * index + 1);
    const double r = option_d_fma(normalized, invc, -1.0);
    const double exponent_double = static_cast<double>(exponent);
    constexpr double ln2hi = 0x1.62e42fefa3800p-1;
    constexpr double ln2lo = 0x1.ef35793c76730p-45;
    const double w = option_d_fma(ln2hi, exponent_double, logc);
    const double hi = w + r;
    const double lo = option_d_fma(ln2lo, exponent_double, (w - hi) + r);
    const double r2 = r * r;
    double y = option_d_fma(option_d_log_poly(0), r2, lo);
    const double r3 = r * r2;
    const double first = option_d_fma(option_d_log_poly(2), r, option_d_log_poly(1));
    const double second = option_d_fma(option_d_log_poly(4), r, option_d_log_poly(3));
    const double polynomial = option_d_fma(second, r2, first);
    y = option_d_fma(r3, polynomial, y);
    return y + hi;
}
