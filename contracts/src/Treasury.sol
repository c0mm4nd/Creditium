// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Multicall} from "@openzeppelin/contracts/utils/Multicall.sol";
import {Oracle} from "./Oracle.sol";
import {KDT} from "./KDTokens.sol";
import {Lending} from "./Lending.sol";

struct Query {
    uint rid;
    uint status; // &0b1 => is_rejected; &0x10 => is_passed; &0x100 => is repaied ...
    address borrower;
    uint256 amount;
    uint256 repaid;
}

contract Treasury is Oracle, KDT {
    mapping(uint => Query) _queries;
    Lending _lending;

    event Borrow(uint rid, address to, uint256 amount);
    event Reject(uint rid);

    event Repay(uint rid);

    constructor(Lending lending) {
        _lending = lending;
    }

    function _sliceUint(
        bytes memory bs,
        uint start
    ) internal pure returns (uint) {
        require(bs.length >= start + 32, "slicing out of range");
        uint x;
        assembly {
            x := mload(add(bs, add(0x20, start)))
        }
        return x;
    }

    function request(
        bytes memory requestContent
    ) public override returns (uint) {
        uint amount = _sliceUint(requestContent, 0);
        uint rid = Oracle.request(requestContent);
        Query memory q = Query(rid, 0x000, msg.sender, amount, 0);
        _queries[rid] = q;

        return rid;
    }

    function repay(uint rid, uint repayAmount) public {
        Query storage q = _queries[rid];

        if (q.amount < q.repaid + repayAmount) {
            // enough for fully repay
            uint amountToBurn = q.amount - q.repaid;
            q.repaid = q.amount;
            super._burn(msg.sender, amountToBurn);
            q.status += 0x100;
        } else {
            q.repaid += repayAmount;
            super._burn(msg.sender, repayAmount);
        }
    }

    function _borrow(Query memory q) private {
        super._mint(q.borrower, q.amount);
        emit Borrow(q.rid, q.borrower, q.amount);
    }

    // alias of the handlerUpdate
    function response(
        uint rid,
        bytes memory respContent
    ) public override onlyRole(ROLE_QUORUM) returns (uint) {
        // Query storage q = _queries[rid];

        // emit EventResponse here
        uint respID = Oracle.response(rid, respContent);
        return respID;
    }

    function approve(
        uint rid,
        uint respID
    ) public override onlyRole(ROLE_QUORUM) returns (bool) {
        bool exit = Oracle.approve(rid, respID);

        if (exit) {
            bytes storage respContent = _responses[rid][respID].respContent;
            bool ok = _sliceUint(respContent, 0) == 1;

            if (ok) {
                _queries[rid].status += 0x010;
                _borrow(_queries[rid]);
            } else {
                _queries[rid].status += 0x001;
                emit Reject(rid);
            }
        }

        return exit;
    }

    function get_model() public view returns (bytes memory) {
        return _lending.get_current_model();
    }
}
