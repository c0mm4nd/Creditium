// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Test.sol";
import "../src/Proving.sol";

contract OracleTest is Test {
    Proving public proving;

    function setUp() public {
        proving = new Proving();
        proving.addQuorum(address(0x7FA9385bE102ac3EAc297483Dd6233D62b3e1496));
        // counter.setNumber(0);
    }

    function testAll() public {
        uint rid = proving.request("bytes");
        uint respID = proving.response(rid, "yes, bytes");
        proving.approve(rid, respID);
    }

    // function testSetNumber(uint256 x) public {
    //     counter.setNumber(x);
    //     assertEq(counter.number(), x);
    // }
}
