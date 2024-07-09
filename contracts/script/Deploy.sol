// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Script.sol";
import {KDT, KDM} from  "../src/KDTokens.sol";
import {Treasury} from  "../src/Treasury.sol";
import {Lending} from  "../src/Lending.sol";
import {Proving} from  "../src/Proving.sol";

contract DeployScript is Script {

    address[] domin_contracts;

    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);

        // Treasury t = new Treasury();
        Lending l = new Lending();
        Proving p = new Proving();

        domin_contracts.push(address(l));
        domin_contracts.push(address(p));

        Treasury kdt = new Treasury(l);
        KDM kdm = new KDM(domin_contracts);

        // t.setToken(kdt);
        l.setGovernToken(kdm);
        p.setGovernToken(kdm);
        
        vm.stopBroadcast();
    }
}
