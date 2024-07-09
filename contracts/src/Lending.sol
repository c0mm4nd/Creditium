// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Oracle} from "./Oracle.sol";

contract Lending is Oracle {
    bytes private _currentPublicModel;
    address[] private _activeAccounts; // TODO: implememnt active account mechinism
    mapping(uint => mapping(address => Handle)) private _handles;

    IERC20 _token;

    struct Handle {
        uint rid;
        address borrower;
        address handler;
        uint round;
    }

    // whitelisted handler start handle the session
    // encryptChanParams contains the factors for the encrypt channel
    event HandleRequest(
        uint indexed rid,
        address indexed handler,
        bytes pubkey,
        bytes initModel
    );
    // model updated by the borrower
    event BorrowerUpdateModel(
        uint indexed rid,
        address indexed borrower,
        address indexed handler,
        bytes model
    );
    // model aggregated by the handler
    event HandlerUpdateModel(
        uint indexed rid,
        address indexed handler,
        uint round,
        bytes model
    );

    // triggered when there's a consensus on the final result
    event PublishModel(uint rid, bytes model);

    // TODO: implement govern revenue
    function setGovernToken(IERC20 token) public onlyRole(ROLE_OWNER) {
        _token = token;
    }

    function request(
        bytes memory provingTxReceipt
    ) public override returns (uint) {
        uint rid = Oracle.request(provingTxReceipt);

        return rid;
    }

    function handle(
        uint rid,
        bytes memory pubKey,
        bytes memory initModel
    ) public onlyRole(ROLE_QUORUM) {
        address borrower = Oracle.getRequestor(rid);
        _handles[rid][msg.sender] = Handle(rid, borrower, msg.sender, 0);

        emit HandleRequest(rid, msg.sender, pubKey, initModel);
    }

    function borrowerUpdate(
        uint rid,
        address[] memory handlers,
        bytes[] memory models
    ) public {
        require(handlers.length == models.length);
        for (uint i = 0; i < handlers.length; i++) {
            Handle memory h = _handles[rid][handlers[i]];
            require(msg.sender == h.borrower); // require caller is borrower

            emit BorrowerUpdateModel(rid, msg.sender, handlers[i], models[i]);
        }
    }

    function handlerUpdate(
        uint[] memory rids,
        bytes[] memory models
    ) public onlyRole(ROLE_QUORUM) {
        require(rids.length == models.length);
        for (uint i = 0; i < rids.length; i++) {
            response(rids[i], models[i]);
        }
    }

    // alias of the handlerUpdate
    function response(
        uint rid,
        bytes memory model
    ) public override onlyRole(ROLE_QUORUM) returns (uint) {
        Handle storage h = _handles[rid][msg.sender];
        // require(msg.sender == r.handler); // require caller is handler

        h.round += 1;
        if (h.round < 10) {
            // expectedRound = 10
            emit HandlerUpdateModel(rid, msg.sender, h.round, model);
            return 0;
        } else {
            // emit EventResponse here
            uint respID = Oracle.response(rid, model);
            return respID;
        }
    }

    function approve(
        uint rid,
        uint respID
    ) public override onlyRole(ROLE_QUORUM) returns (bool) {
        bool exit = Oracle.approve(rid, respID);

        if (exit) {
            _currentPublicModel = _responses[rid][respID].respContent;
            emit PublishModel(rid, _currentPublicModel);

            // delete _handles[rid];
        }

        return exit;
    }

    function get_current_model() public view returns (bytes memory) {
        return _currentPublicModel;
    }
}
