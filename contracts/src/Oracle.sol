// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {Governor} from "@openzeppelin/contracts/governance/Governor.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Counters} from "@openzeppelin/contracts/utils/Counters.sol";
import {Multicall} from "@openzeppelin/contracts/utils/Multicall.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

// Add the library methods
using EnumerableSet for EnumerableSet.AddressSet;
using Counters for Counters.Counter;

struct Request {
    uint id;
    address user;
    bytes reqContent; // callback to run the real handlers
}

struct Response {
    // The Request
    uint id;
    // The response part
    address respQuorum;
    bytes respContent;
}

contract Oracle is AccessControl, Multicall {
    bytes32 public constant ROLE_OWNER = DEFAULT_ADMIN_ROLE;
    bytes32 public constant ROLE_QUORUM = keccak256("QUORUM");

    Counters.Counter private _counter; // can be 0 -> max256

    mapping(uint => Request) _requests;
    mapping(uint => Response[]) _responses;
    mapping(uint => mapping(uint => EnumerableSet.AddressSet)) _approvers;

    event EventRequest(uint indexed id, address indexed user, bytes reqContent);
    event EventResponse(
        uint indexed id,
        uint indexed respID,
        address indexed respQuorum,
        bytes respContent
    );
    event EventDone(uint indexed id, uint indexed respID, bytes respContent);

    constructor() {
        _grantRole(ROLE_OWNER, msg.sender);
    }

    function getRequestor(uint rid) public view returns (address) {
        return _requests[rid].user;
    }

    function request(bytes memory content) public virtual returns (uint) {
        uint rid = _counter.current();
        _requests[rid] = Request(rid, msg.sender, content);

        _counter.increment();

        emit EventRequest(rid, msg.sender, content);
        return rid;
    }

    function response(
        uint rid,
        bytes memory responseContent
    ) public virtual onlyRole(ROLE_QUORUM) returns (uint) {
        _responses[rid].push(Response(rid, msg.sender, responseContent)); // use tx.origin to restrict the oracle respQuorum being EOA
        uint respID = _responses[rid].length - 1;
        require(_approvers[rid][respID].add(msg.sender));

        emit EventResponse(rid, respID, msg.sender, responseContent);
        return respID;
    }

    function approve(
        uint rid,
        uint respID
    ) public virtual onlyRole(ROLE_QUORUM) returns (bool exit) {
        if (_approvers[rid][respID].length() >= 1) {
            // TODO: change the magic number to a proper value in prod env
            _done(rid, respID);
            return true;
        }

        return false;
    }

    function _done(uint rid, uint respID) private onlyRole(ROLE_QUORUM) {
        // r.callbackFn(respID);
        emit EventDone(rid, respID, _responses[rid][respID].respContent);

        // delete _responses[rid];
        // delete _requests[rid];
        // delete _approvers[rid];
    }

    function addQuorum(address quorum) public onlyRole(ROLE_OWNER) {
        grantRole(ROLE_QUORUM, quorum);
    }

    function removeQuorum(address quorum) public onlyRole(ROLE_OWNER) {
        renounceRole(ROLE_QUORUM, quorum);
    }

    //TODO: add voting to addQuorum
}
