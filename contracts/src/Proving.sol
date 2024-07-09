// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Oracle} from "./Oracle.sol";

contract Proving is Oracle {
    IERC20 _token;

    constructor() {}

    function request(
        bytes memory encryptedPersonalID
    ) public override returns (uint) {
        uint rid = Oracle.request(encryptedPersonalID);
        return rid;
    }

    function response(
        uint rid,
        bytes memory encryptedPersonalDetails
    ) public override returns (uint) {
        uint respID = Oracle.response(rid, encryptedPersonalDetails);
        return respID;
    }

    function approve(uint rid, uint respID) public override returns (bool) {
        bool exit = Oracle.approve(rid, respID);

        return exit;
    }

    // TODO: implement govern revenue
    function setGovernToken(IERC20 token) public onlyRole(ROLE_OWNER) {
        _token = token;
    }
}
